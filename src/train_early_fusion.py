"""
train_early_fusion.py
======================
EARLY FUSION VERSI PENUH -- semua blok fitur digabung (concat) jadi SATU vektor
per hari, lalu dimakan SATU XGBoost. Lanjutan langsung dari train_combined.py:
pertanyaannya "apakah embedding kontekstual / topik / NER menambah informasi DI
ATAS model utama (harga + TF-IDF + LM)?". Lihat docs/keputusan.md #46-#56.

BLOK FITUR (urutan kolom selalu sama: harga, lm, tfidf, emb, topic, ner, turunan)
--------------------------------------------------------------------------
harga   : PRICE_FEATURES (dataset_split.py)                          10 kolom
lm      : LM_FEATURES (dataset_split.py)                              3 kolom
tfidf   : TF-IDF -> TruncatedSVD, persis train_combined.py           20 kolom
emb     : embedding harian (FinBERT / RoBERTa / Qwen x headline /
          fulltext) -> StandardScaler -> PCA                         20 kolom
topic   : LDA 15 topik yang DI-FIT ULANG DI ARTIKEL TRAIN SAJA,
          rata-rata distribusi topik per hari                        15 kolom
ner     : fitur entitas DINORMALISASI per artikel (grup + entitas
          kunci yang ditetapkan dari domain knowledge)               21 kolom
turunan : news novelty, topic entropy, topic shift (ablasi saja)      3 kolom

SEMUA transformasi yang belajar dari data (TF-IDF, SVD, scaler, PCA, LDA)
di-fit di TRAIN saja. Semua XGBoost di-tuning pakai tune_xgb + XGB_GRID yang
sama dengan Tugas 2, dipilih via akurasi VALID.

PROTOKOL: model final = run "full" (F1 + emb + topic + ner) dengan akurasi
VALID tertinggi di antara 6 embedding. HANYA model final yang diklaim di test;
run lain dilaporkan sebagai analisis.

Jalankan dari folder src/:  python train_early_fusion.py
"""

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest
from sklearn.decomposition import PCA, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

import config as C
from dataset_split import LM_FEATURES, PRICE_FEATURES, build_dataset, log_split_summary, tune_xgb

# tokenizer + hyperparameter LDA diambil dari modul topic modelling supaya
# LDA train-only di sini SAMA PERSIS resepnya dengan LDA yang sudah dievaluasi
# (coherence dsb.) -- bedanya cuma data yang dipakai fit.
from topic_modelling.topic_models import (
    LDA_MAX_DF, LDA_MAX_FEATURES, LDA_MAX_ITER, LDA_MIN_DF, N_TOPICS, tokenize_text,
)

# --- konfigurasi -------------------------------------------------------------
EMB_MODELS = ("finbert", "roberta", "qwen")
TEXT_TYPES = ("headline", "fulltext")
EMBEDDINGS = [f"{m}_{t}" for m in EMB_MODELS for t in TEXT_TYPES]

N_PCA_COMPONENTS = 20          # sama dengan N_SVD_COMPONENTS TF-IDF (Keputusan 40)
ALT_AGGREGATION = "A2_exp_decay_hl3d"   # satu-satunya agregasi alternatif (ablasi)

# TF-IDF: identik dengan train_combined.py
MIN_DF, MAX_DF, NGRAM_RANGE, MAX_FEATURES, N_SVD_COMPONENTS = 5, 0.8, (1, 2), 5000, 20

# NER: entitas kunci ditetapkan dari domain knowledge (BUKAN dari hasil valid)
NER_KEY_ENTITIES = [
    "country_united_states_mentions", "country_china_mentions", "country_russia_mentions",
    "country_ukraine_mentions", "country_israel_mentions", "country_iran_mentions",
    "country_indonesia_mentions",
    "central_bank_federal_reserve_mentions", "central_bank_bank_indonesia_mentions",
    "event_monetary_policy_count", "event_sanctions_count", "event_trade_policy_count",
    "event_currency_fx_count", "event_energy_commodity_count",
]
NER_GROUPS = ("country", "central_bank", "economic_event")

NOVELTY_WINDOW = 30      # hari berberita sebelumnya
TOPIC_SHIFT_WINDOW = 7   # hari berberita sebelumnya
N_BOOTSTRAP = 5000
SEEDS = (0, 1, 2, 3, 4)
RANDOM_STATE = 42

HEAD_VS_FULL_DIR = C.DATA_PROCESSED / "head_vs_full"
TEMPORAL_DIR = C.DATA_PROCESSED / "temporal_aggregation"
TOPIC_DOCS_CSV = C.DATA_PROCESSED / "topic_modelling" / "input_documents.csv"
NER_DAILY_CSV = C.DATA_PROCESSED / "ner" / "daily_entity_features.csv"

OUT_DIR = C.DATA_PROCESSED / "early_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LDA_CACHE_CSV = OUT_DIR / "lda_train_only_daily_topic_features.csv"
LDA_CACHE_META = OUT_DIR / "lda_train_only_metadata.json"

LOG_DIR = C.ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = LOG_DIR / "train_early_fusion_report.txt"
MODEL_DIR = C.ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "valid", "test")


# =============================================================================
# util
# =============================================================================
def mcnemar(y_true, pred_a, pred_b):
    """Uji McNemar exact (sama dengan train_combined.py)."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_only = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    b_only = int(((pred_a != y_true) & (pred_b == y_true)).sum())
    p = binomtest(a_only, a_only + b_only).pvalue if a_only + b_only else 1.0
    return a_only, b_only, p


def holm(pvalues):
    """p-value terkoreksi Holm (step-down), urutan output = urutan input."""
    order = np.argsort(pvalues)
    m, adjusted, running = len(pvalues), np.empty(len(pvalues)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def bootstrap_ci(y_true, pred_a, pred_b=None, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """
    CI 95% bootstrap atas HARI TEST. Tanpa pred_b: CI akurasi A. Dengan pred_b:
    CI selisih akurasi A - B (berpasangan: hari yang sama di-resample untuk keduanya).
    """
    y_true = np.asarray(y_true)
    hit = (np.asarray(pred_a) == y_true).astype(float)
    if pred_b is not None:
        hit = hit - (np.asarray(pred_b) == y_true).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hit), size=(n_boot, len(hit)))
    lo, hi = np.percentile(hit[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def by_split(df, cols):
    """dict split -> matriks numpy kolom `cols`."""
    return {s: df.loc[df["split"] == s, cols].to_numpy(dtype=float) for s in SPLITS}


def merge_on_date(days, feats, name):
    """Left-join fitur harian ke hari resmi; WAJIB lengkap (tidak boleh ada hari kosong)."""
    feats = feats.copy()
    feats["date"] = pd.to_datetime(feats["date"])
    assert feats["date"].is_unique, f"{name}: tanggal duplikat"
    out = days[["date", "split"]].merge(feats, on="date", how="left", indicator=True)
    missing = int((out["_merge"] != "both").sum())
    assert missing == 0, f"{name}: {missing} hari resmi tidak punya fitur"
    return out.drop(columns="_merge")


# =============================================================================
# blok fitur -- tiap fungsi mengembalikan {"cols": [...], "train"/"valid"/"test": ndarray}
# =============================================================================
def block_tfidf(days, log):
    text = {s: days.loc[days["split"] == s, "text_norm_concat"].fillna("") for s in SPLITS}
    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES)
    X = {"train": vectorizer.fit_transform(text["train"])}
    X["valid"], X["test"] = vectorizer.transform(text["valid"]), vectorizer.transform(text["test"])
    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=RANDOM_STATE)
    block = {"train": svd.fit_transform(X["train"])}
    block["valid"], block["test"] = svd.transform(X["valid"]), svd.transform(X["test"])
    block["cols"] = [f"tfidf_svd_{i}" for i in range(N_SVD_COMPONENTS)]
    block["fitted"] = {"vectorizer": vectorizer, "svd": svd}
    log(f"[tfidf] vocab {len(vectorizer.vocabulary_):,} -> {N_SVD_COMPONENTS} komponen SVD "
        f"({svd.explained_variance_ratio_.sum()*100:.1f}% varians)")
    return block


def embedding_path(name, aggregation):
    model, text_type = name.split("_")
    if aggregation == "A0":
        return HEAD_VS_FULL_DIR / text_type / f"{name}_features.csv"
    return TEMPORAL_DIR / name / f"{name}_{aggregation}.csv"


def load_daily_embedding(name, aggregation="A0"):
    df = pd.read_csv(embedding_path(name, aggregation))
    prefix = name.split("_")[0] + "_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    assert len(cols) >= 100, f"{name}: kolom embedding tidak ditemukan"
    df["date"] = pd.to_datetime(df["date"])
    return df[["date"] + cols], cols


def block_embedding(days, name, log, aggregation="A0"):
    """Embedding harian -> StandardScaler -> PCA(20). Scaler & PCA fit di TRAIN saja."""
    raw, cols = load_daily_embedding(name, aggregation)
    merged = merge_on_date(days, raw, f"emb:{name}")
    X = by_split(merged, cols)
    for s in SPLITS:  # hari resmi semuanya berberita -> tidak boleh ada vektor nol / NaN
        assert np.isfinite(X[s]).all(), f"emb:{name} {s}: ada NaN"
        assert (np.abs(X[s]).sum(axis=1) > 0).all(), f"emb:{name} {s}: ada vektor nol"
    scaler = StandardScaler().fit(X["train"])
    pca = PCA(n_components=N_PCA_COMPONENTS, svd_solver="full", random_state=RANDOM_STATE)
    pca.fit(scaler.transform(X["train"]))
    block = {s: pca.transform(scaler.transform(X[s])) for s in SPLITS}
    tag = name if aggregation == "A0" else f"{name}_{aggregation}"
    block["cols"] = [f"emb_{tag}_pc{i:02d}" for i in range(N_PCA_COMPONENTS)]
    block["fitted"] = {"scaler": scaler, "pca": pca}
    log(f"[emb:{tag}] {len(cols)} dim -> {N_PCA_COMPONENTS} PC "
        f"({pca.explained_variance_ratio_.sum()*100:.1f}% varians train)")
    return block


def lda_train_only_daily(train_end, log):
    """
    Distribusi topik harian dari LDA yang di-fit HANYA di artikel dengan
    target_date <= akhir train (bukan transduktif). Hasil di-cache; cache
    dipakai ulang hanya kalau konfigurasinya identik.
    """
    config = {
        "train_end": str(train_end.date()), "n_topics": N_TOPICS, "max_features": LDA_MAX_FEATURES,
        "min_df": LDA_MIN_DF, "max_df": LDA_MAX_DF, "max_iter": LDA_MAX_ITER, "random_state": RANDOM_STATE,
    }
    if LDA_CACHE_CSV.exists() and LDA_CACHE_META.exists():
        meta = json.loads(LDA_CACHE_META.read_text(encoding="utf-8"))
        if meta.get("config") == config:
            log(f"[topic] pakai cache LDA train-only: {LDA_CACHE_CSV.name} "
                f"({meta['n_train_docs']:,} artikel train)")
            return pd.read_csv(LDA_CACHE_CSV, parse_dates=["date"])

    docs = pd.read_csv(TOPIC_DOCS_CSV, usecols=["target_date", "text_norm"])
    docs["target_date"] = pd.to_datetime(docs["target_date"])
    docs["text_norm"] = docs["text_norm"].fillna("").astype(str)
    is_train = docs["target_date"] <= train_end

    vectorizer = CountVectorizer(
        tokenizer=tokenize_text, token_pattern=None, lowercase=True,
        max_features=LDA_MAX_FEATURES, min_df=LDA_MIN_DF, max_df=LDA_MAX_DF,
    )
    dtm_train = vectorizer.fit_transform(docs.loc[is_train, "text_norm"])
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS, max_iter=LDA_MAX_ITER, learning_method="batch",
        random_state=RANDOM_STATE, n_jobs=-1, evaluate_every=-1,
    )
    t0 = time.time()
    lda.fit(dtm_train)
    theta = lda.transform(vectorizer.transform(docs["text_norm"]))  # baris sudah berjumlah 1
    topic_cols = [f"topic_{i:02d}_mean" for i in range(N_TOPICS)]
    daily = (
        pd.DataFrame(theta, columns=topic_cols).assign(date=docs["target_date"].to_numpy())
        .groupby("date", as_index=False)[topic_cols].mean().sort_values("date")
    )
    daily.to_csv(LDA_CACHE_CSV, index=False)
    feature_names = vectorizer.get_feature_names_out()
    top_words = {f"topic_{i:02d}": feature_names[comp.argsort()[-10:][::-1]].tolist()
                 for i, comp in enumerate(lda.components_)}
    LDA_CACHE_META.write_text(json.dumps({
        "config": config, "n_train_docs": int(is_train.sum()), "n_all_docs": int(len(docs)),
        "vocab_size": int(len(feature_names)), "top_words": top_words,
    }, indent=2), encoding="utf-8")
    log(f"[topic] LDA di-fit di {int(is_train.sum()):,} artikel train (dari {len(docs):,}), "
        f"vocab {len(feature_names):,}, {time.time()-t0:.0f} dtk")
    return daily


def block_topic(days, topic_daily):
    cols = [c for c in topic_daily.columns if c.startswith("topic_") and c.endswith("_mean")]
    block = by_split(merge_on_date(days, topic_daily, "topic"), cols)
    block["cols"] = cols
    return block


def block_ner(days):
    """
    Fitur NER per ARTIKEL (dibagi n_news) -- kolom mentahnya berupa JUMLAH
    harian yang ikut membesar dengan volume berita (masalah yang sama dengan
    lm_*_sum, Keputusan 39).
    """
    ner = pd.read_csv(NER_DAILY_CSV)
    merged = merge_on_date(days, ner, "ner")
    n_news = days["n_news"].to_numpy(dtype=float)
    assert (n_news > 0).all()
    out = merged[["date", "split"]].copy()
    out["ner_entities_per_article"] = merged["total_entity_mentions"].to_numpy() / n_news
    for g in NER_GROUPS:
        out[f"ner_share_articles_with_{g}"] = merged[f"n_articles_with_{g}"].to_numpy() / n_news
        out[f"ner_{g}_mentions_per_article"] = merged[f"daily_{g}_mentions"].to_numpy() / n_news
    for col in NER_KEY_ENTITIES:
        out["ner_" + col.replace("_mentions", "").replace("_count", "") + "_per_article"] = merged[col].to_numpy() / n_news
    cols = [c for c in out.columns if c.startswith("ner_")]
    block = by_split(out.fillna(0.0), cols)
    block["cols"] = cols
    return block


def block_derived(days, emb_name, topic_daily):
    """
    Fitur turunan berdimensi kecil -- semuanya hanya memakai hari <= t:
    - news_novelty : 1 - cosine(embedding hari t, rata-rata NOVELTY_WINDOW hari berberita sebelumnya)
    - topic_entropy: entropi distribusi topik harian (berita fokus vs tersebar)
    - topic_shift  : jarak Jensen-Shannon distribusi topik hari t vs rata-rata
                     TOPIC_SHIFT_WINDOW hari berberita sebelumnya
    """
    raw, cols = load_daily_embedding(emb_name)
    raw = raw.sort_values("date")
    E = raw[cols].to_numpy(dtype=float)
    has_news = np.abs(E).sum(axis=1) > 0
    E_news = pd.DataFrame(E[has_news])
    prev = E_news.rolling(NOVELTY_WINDOW, min_periods=1).mean().shift(1).to_numpy()
    cur = E_news.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (cur * prev).sum(axis=1) / (np.linalg.norm(cur, axis=1) * np.linalg.norm(prev, axis=1))
    novelty = pd.DataFrame({"date": raw.loc[has_news, "date"].to_numpy(), "news_novelty": 1.0 - cos})

    tcols = [c for c in topic_daily.columns if c.endswith("_mean")]
    T = topic_daily.sort_values("date")
    P = T[tcols].to_numpy(dtype=float)
    P = P / P.sum(axis=1, keepdims=True)
    P_prev = pd.DataFrame(P).rolling(TOPIC_SHIFT_WINDOW, min_periods=1).mean().shift(1).to_numpy()
    shift = np.array([jensenshannon(p, q) if np.isfinite(q).all() else np.nan for p, q in zip(P, P_prev)])
    topic_feats = pd.DataFrame({
        "date": T["date"].to_numpy(),
        "topic_entropy": -(P * np.log(np.clip(P, 1e-12, None))).sum(axis=1),
        "topic_shift": shift,
    })
    feats = novelty.merge(topic_feats, on="date", how="outer")
    cols_out = ["news_novelty", "topic_entropy", "topic_shift"]
    block = by_split(merge_on_date(days, feats, "turunan"), cols_out)  # NaN (hari pertama) dibiarkan: XGBoost menanganinya
    block["cols"] = cols_out
    return block


# =============================================================================
def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    def quiet(msg=""):  # log tune_xgb: masuk laporan, tidak membanjiri terminal
        lines.append("    " + str(msg))

    t_start = time.time()
    train, valid, test = build_dataset(required_cols=PRICE_FEATURES + LM_FEATURES)
    log_split_summary(log, train, valid, test)
    assert (len(train), len(valid), len(test)) == (772, 165, 167), "himpunan hari berbeda dari split resmi"
    days = pd.concat([p.assign(split=s) for s, p in zip(SPLITS, (train, valid, test))], ignore_index=True)
    train_end = train["date"].max()

    y = {s: days.loc[days["split"] == s, "volatility_class"].astype(str).to_numpy() for s in SPLITS}
    labels = sorted(set(y["train"]))
    encoder = LabelEncoder().fit(y["train"])
    y_train_enc = encoder.transform(y["train"])
    naive = {s: days.loc[days["split"] == s, "lag1_volatility_class"].astype(str).to_numpy() for s in SPLITS}

    # --- bangun semua blok ---------------------------------------------------
    log("=" * 72)
    log("MEMBANGUN BLOK FITUR (semua transformasi di-fit di TRAIN saja)")
    log("=" * 72)
    blocks = {
        "harga": {**by_split(days, PRICE_FEATURES), "cols": list(PRICE_FEATURES)},
        "lm": {**by_split(days.fillna({c: 0 for c in LM_FEATURES}), LM_FEATURES), "cols": list(LM_FEATURES)},
        "tfidf": block_tfidf(days, log),
    }
    for name in EMBEDDINGS:
        blocks[f"emb:{name}"] = block_embedding(days, name, log)
    topic_daily = lda_train_only_daily(train_end, log)
    blocks["topic"] = block_topic(days, topic_daily)
    blocks["ner"] = block_ner(days)
    log(f"[topic] {len(blocks['topic']['cols'])} kolom | [ner] {len(blocks['ner']['cols'])} kolom: {blocks['ner']['cols']}")
    log()

    BASE = ["harga", "lm", "tfidf"]
    results, preds, fitted = [], {}, {}

    def block_kind(b):
        return b.split(":")[0]

    def run(run_id, group, block_names):
        X = {s: np.hstack([blocks[b][s] for b in block_names]) for s in SPLITS}
        feat_blocks = [block_kind(b) for b in block_names for _ in blocks[b]["cols"]]
        lines.append(f"[{run_id}] blok={block_names} n_fitur={X['train'].shape[1]}")
        clf, pred_valid, pred_test = tune_xgb(X["train"], y_train_enc, X["valid"], y["valid"], X["test"], encoder, quiet)
        importance = {}
        for kind, imp in zip(feat_blocks, clf.feature_importances_):
            importance[kind] = importance.get(kind, 0.0) + float(imp)
        params = {k: clf.get_params()[k] for k in ("n_estimators", "max_depth", "learning_rate")}
        row = {
            "run_id": run_id, "group": group, "blocks": " + ".join(block_names), "n_features": X["train"].shape[1],
            "acc_valid": accuracy_score(y["valid"], pred_valid), "acc_test": accuracy_score(y["test"], pred_test),
            "macro_f1_valid": f1_score(y["valid"], pred_valid, average="macro"),
            "macro_f1_test": f1_score(y["test"], pred_test, average="macro"),
            **params, "block_importance": json.dumps({k: round(v, 4) for k, v in importance.items()}),
        }
        results.append(row)
        preds[run_id] = {"valid": pred_valid, "test": pred_test}
        fitted[run_id] = {"clf": clf, "X": X, "params": params, "block_names": block_names}
        log(f"  {run_id:<44} fitur={row['n_features']:>4}  valid={row['acc_valid']:.3f}  test={row['acc_test']:.3f}  "
            f"F1test={row['macro_f1_test']:.3f}")
        return row

    # --- 1. baseline ---------------------------------------------------------
    log("=" * 72)
    log("EKSPERIMEN (akurasi; semua XGBoost + XGB_GRID yang sama, dipilih via VALID)")
    log("=" * 72)
    log(f"  {'naive persistence':<44} fitur=   0  valid={accuracy_score(y['valid'], naive['valid']):.3f}  "
        f"test={accuracy_score(y['test'], naive['test']):.3f}  "
        f"F1test={f1_score(y['test'], naive['test'], average='macro'):.3f}")
    run("F0_harga", "baseline", ["harga"])
    run("F1_harga+lm+tfidf", "baseline", BASE)
    log()

    # --- 2. blok non-embedding di atas F1 -------------------------------------
    run("F1+topic", "tanpa_embedding", BASE + ["topic"])
    run("F1+ner", "tanpa_embedding", BASE + ["ner"])
    run("F1+topic+ner", "tanpa_embedding", BASE + ["topic", "ner"])
    log()

    # --- 3. per embedding (opsi alpha) ----------------------------------------
    for name in EMBEDDINGS:
        e = f"emb:{name}"
        run(f"harga+lm+{name}", "emb_vs_tfidf", ["harga", "lm", e])          # base A: embedding menggantikan TF-IDF
        run(f"F1+{name}", "f1_plus_emb", BASE + [e])
        run(f"F1+{name}+topic", "f1_plus_emb_topic", BASE + [e, "topic"])
        run(f"F1+{name}+ner", "f1_plus_emb_ner", BASE + [e, "ner"])
        run(f"FULL_{name}", "full", BASE + [e, "topic", "ner"])
        log()

    # --- 4. kitchen sink (opsi beta) -------------------------------------------
    run("KITCHEN_SINK_6emb", "kitchen_sink", BASE + [f"emb:{n}" for n in EMBEDDINGS] + ["topic", "ner"])
    log()

    # --- 5. pilih model final: FULL dengan akurasi VALID tertinggi -------------
    full = [r for r in results if r["group"] == "full"]
    # tie-break: macro F1 valid, lalu urutan EMBEDDINGS (max() mengambil yang pertama)
    final = max(full, key=lambda r: (round(r["acc_valid"], 10), round(r["macro_f1_valid"], 10)))
    final_id = final["run_id"]
    final_emb = final_id.replace("FULL_", "")
    e = f"emb:{final_emb}"
    log("=" * 72)
    log(f"MODEL FINAL (dipilih via akurasi VALID di antara 6 run FULL): {final_id}")
    log("=" * 72)
    log(f"Hyperparameter: {fitted[final_id]['params']}")
    log(f"Akurasi valid : {final['acc_valid']:.3f}")
    log(f"Akurasi test  : {final['acc_test']:.3f}")
    log()
    log("Classification report (TEST set):")
    log(classification_report(y["test"], preds[final_id]["test"], zero_division=0))
    log("Confusion matrix TEST (baris=aktual, kolom=prediksi), label urut " + str(labels))
    log(confusion_matrix(y["test"], preds[final_id]["test"], labels=labels))
    log()
    log(f"Total feature importance per blok: {final['block_importance']}")
    log()

    # --- 6. ablasi model final -------------------------------------------------
    log("ABLASI MODEL FINAL (leave-one-block-out; baris yang sudah dijalankan dirujuk ulang)")
    full_blocks = BASE + [e, "topic", "ner"]
    ablation_ref = {"emb": "F1+topic+ner", "topic": f"F1+{final_emb}+ner", "ner": f"F1+{final_emb}+topic"}
    ablation_rows = []
    for drop in ("harga", "lm", "tfidf"):
        r = run(f"FINAL-tanpa_{drop}", "ablasi_final", [b for b in full_blocks if b != drop])
        ablation_rows.append((drop, r))
    by_id = {r["run_id"]: r for r in results}
    for drop, ref in ablation_ref.items():
        ablation_rows.append((drop, by_id[ref]))
    log()
    log(f"  {'blok dibuang':<14} {'valid':>7} {'test':>7} {'delta test vs FINAL':>22}")
    for drop, r in ablation_rows:
        log(f"  {drop:<14} {r['acc_valid']:>7.3f} {r['acc_test']:>7.3f} {r['acc_test']-final['acc_test']:>+22.3f}")
    log()

    log(f"ABLASI AGREGASI TEMPORAL: embedding final dengan {ALT_AGGREGATION} (default: A0 daily mean)")
    blocks[f"emb:{final_emb}@alt"] = block_embedding(days, final_emb, log, aggregation=ALT_AGGREGATION)
    run(f"FINAL_agg_{ALT_AGGREGATION}", "ablasi_agregasi", BASE + [f"emb:{final_emb}@alt", "topic", "ner"])
    log()

    log("ABLASI FITUR TURUNAN: FINAL + news_novelty + topic_entropy + topic_shift")
    blocks["turunan"] = block_derived(days, final_emb, topic_daily)
    run("FINAL+turunan", "ablasi_turunan", full_blocks + ["turunan"])
    log()

    # --- 7. signifikansi & ketidakpastian ---------------------------------------
    log("=" * 72)
    log("UJI SIGNIFIKANSI & KETIDAKPASTIAN (TEST, 167 hari) -- hanya untuk MODEL FINAL")
    log("=" * 72)
    yt = y["test"]
    comparisons = [
        ("FINAL vs F1 (harga+lm+tfidf)", preds["F1_harga+lm+tfidf"]["test"]),
        ("FINAL vs F0 (harga saja)", preds["F0_harga"]["test"]),
        ("FINAL vs naive persistence", naive["test"]),
    ]
    stats = [mcnemar(yt, preds[final_id]["test"], other) for _, other in comparisons]
    p_holm = holm([s[2] for s in stats])
    log("McNemar exact ('FINAL saja benar / pembanding saja benar'), p mentah dan p terkoreksi Holm:")
    significance = []
    for (name, other), (a_only, b_only, p), ph in zip(comparisons, stats, p_holm):
        lo, hi = bootstrap_ci(yt, preds[final_id]["test"], other)
        log(f"  {name:<30}: {a_only:>3} / {b_only:<3} hari  p = {p:.3f}  p_Holm = {ph:.3f}  "
            f"{'SIGNIFIKAN' if ph < 0.05 else 'tidak signifikan'}  | selisih akurasi CI95% bootstrap [{lo:+.3f}, {hi:+.3f}]")
        significance.append({"comparison": name, "final_only": a_only, "other_only": b_only,
                             "p_mcnemar": p, "p_holm": float(ph), "diff_ci95": [lo, hi]})
    log()
    log(f"CI 95% bootstrap akurasi test ({N_BOOTSTRAP:,} resample hari):")
    acc_ci = {}
    for name, p in (("naive persistence", naive["test"]), ("F0_harga", preds["F0_harga"]["test"]),
                    ("F1_harga+lm+tfidf", preds["F1_harga+lm+tfidf"]["test"]), (final_id, preds[final_id]["test"])):
        lo, hi = bootstrap_ci(yt, p)
        acc_ci[name] = [lo, hi]
        log(f"  {name:<30}: {accuracy_score(yt, p):.3f}  [{lo:.3f}, {hi:.3f}]")
    log()

    log(f"Multi-seed (hyperparameter terpilih, random_state = {SEEDS}):")
    seed_stats = {}
    for run_id in ("F1_harga+lm+tfidf", final_id):
        f = fitted[run_id]
        accs = []
        for seed in SEEDS:
            clf = XGBClassifier(**f["params"], eval_metric="mlogloss", random_state=seed)
            clf.fit(f["X"]["train"], y_train_enc)
            accs.append(accuracy_score(yt, encoder.inverse_transform(clf.predict(f["X"]["test"]))))
        seed_stats[run_id] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "accs": accs}
        log(f"  {run_id:<30}: akurasi test {np.mean(accs):.3f} +/- {np.std(accs):.3f}  {[round(a, 3) for a in accs]}")
    if all(s["std"] == 0 for s in seed_stats.values()):
        log("  -> std = 0: XGBoost tanpa subsampling bersifat deterministik, jadi seed tidak")
        log("     menambah variasi. Ketidakpastian dilaporkan lewat CI bootstrap di atas.")
    log()

    # --- 8. simpan -----------------------------------------------------------------
    res = pd.DataFrame(results)
    res["is_final"] = res["run_id"] == final_id
    res.to_csv(OUT_DIR / "early_fusion_results.csv", index=False)
    pred_df = days.loc[days["split"] == "test", ["date"]].reset_index(drop=True)
    pred_df["y_true"], pred_df["naive"] = yt, naive["test"]
    pred_df = pd.concat([pred_df, pd.DataFrame({k: v["test"] for k, v in preds.items()})], axis=1)
    pred_df.to_csv(OUT_DIR / "early_fusion_test_predictions.csv", index=False)

    metadata = {
        "target": "volatility_class (low/medium/high)", "split_sizes": {s: int(len(y[s])) for s in SPLITS},
        "train_end": str(train_end.date()), "embeddings": EMBEDDINGS, "default_aggregation": "A0 daily mean",
        "alt_aggregation": ALT_AGGREGATION, "n_pca_components": N_PCA_COMPONENTS,
        "n_svd_components_tfidf": N_SVD_COMPONENTS, "topic_model": "LDA train-only (15 topik)",
        "ner_features": blocks["ner"]["cols"], "classifier": "XGBoost + tune_xgb (XGB_GRID, pilih via akurasi valid)",
        "final_selection_rule": "run FULL dengan akurasi valid tertinggi; tie-break macro F1 valid lalu urutan EMBEDDINGS",
        "final_run": final_id, "final_params": fitted[final_id]["params"],
        "final_block_importance": json.loads(final["block_importance"]),
        "significance": significance, "accuracy_ci95_bootstrap": acc_ci, "multi_seed": seed_stats,
        "n_runs": len(results),
    }
    (OUT_DIR / "early_fusion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    feature_names = [c for b in fitted[final_id]["block_names"] for c in blocks[b]["cols"]]
    joblib.dump({
        "clf": fitted[final_id]["clf"], "label_encoder": encoder, "feature_names": feature_names,
        "block_names": fitted[final_id]["block_names"], "tfidf": blocks["tfidf"]["fitted"],
        "embedding": {"name": final_emb, **blocks[e]["fitted"]},
    }, MODEL_DIR / "early_fusion_final.pkl")

    log(f"Tersimpan: {OUT_DIR / 'early_fusion_results.csv'}")
    log(f"           {OUT_DIR / 'early_fusion_test_predictions.csv'}")
    log(f"           {OUT_DIR / 'early_fusion_metadata.json'}")
    log(f"           {MODEL_DIR / 'early_fusion_final.pkl'}")
    log(f"Total {len(results)} run, {time.time()-t_start:.0f} dtk")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLaporan tersimpan: {REPORT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
