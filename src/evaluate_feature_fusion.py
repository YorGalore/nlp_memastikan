from __future__ import annotations

import ast
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from dataset_split import PRICE_FEATURES, TRAIN_FRAC, VALID_FRAC, build_dataset

# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ("finbert", "roberta", "qwen")
TEXT_TYPES = ("headline", "fulltext")
CLASS_ORDER = ("low", "medium", "high")
RANDOM_STATE = 42
LOGISTIC_C = 1.0
# Reuse the canonical market features and chronological split from
# dataset_split.py. Every experiment must use the same target days, labels,
# cutoffs, and 70/15/15 boundaries.
JISDOR_FEATURES = PRICE_FEATURES

# LDA is the default because the project's topic evaluation showed full
# document coverage. Change to bertopic/dtm/top2vec if that is the chosen model.
TOPIC_MODEL = "lda"

EXPERIMENTS = {
    "E0_jisdor_price_only": ("jisdor",),
    "E1_jisdor_plus_embedding": ("jisdor", "embedding"),
    "E2_jisdor_plus_embedding_topic": ("jisdor", "embedding", "topic"),
    "E3_jisdor_plus_embedding_ner": ("jisdor", "embedding", "ner"),
    "E4_jisdor_plus_embedding_topic_ner": ("jisdor", "embedding", "topic", "ner"),
    "E5_jisdor_plus_embedding_temporal": ("jisdor", "embedding", "temporal"),
    "E6_jisdor_plus_embedding_topic_temporal": ("jisdor", "embedding", "topic", "temporal"),
    "E7_jisdor_plus_embedding_ner_temporal": ("jisdor", "embedding", "ner", "temporal"),
    "E8_jisdor_plus_embedding_topic_ner_temporal": ("jisdor", "embedding", "topic", "ner", "temporal"),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("feature_fusion_evaluation")


def find_project_root() -> Path:
    """Find repository root from script location or current working directory."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if (parent / "data" / "processed" / "aligned_daily.csv").exists():
            return parent
    raise FileNotFoundError(
        "Tidak menemukan data/processed/aligned_daily.csv. Jalankan dari root repository."
    )


ROOT_DIR = find_project_root()
DATA_DIR = ROOT_DIR / "data" / "processed"
TARGET_FILE = DATA_DIR / "aligned_daily.csv"
EMBEDDING_DIR = DATA_DIR / "head_vs_full"
TOPIC_FILE = DATA_DIR / "topic_modelling" / TOPIC_MODEL / "daily_topic_features.csv"
NER_FILE = DATA_DIR / "ner" / "daily_entity_features.csv"
# Chosen from the completed JISDOR-aware temporal smoke evaluation. Change this
# explicit path after a full A0--A4 temporal sweep identifies a better variant.
TEMPORAL_FILE = DATA_DIR / "temporal_aggregation" / "doc2vec_fulltext" / "doc2vec_fulltext_A0_daily_mean.csv"
OUTPUT_DIR = DATA_DIR / "feature_fusion_evaluation"


def parse_vector(value: Any) -> list[float] | None:
    """Parse an embedding vector stored as a list/JSON/Python literal string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return None

    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            cleaned = raw.strip("[]() ")
            if not cleaned:
                return None
            values = np.fromstring(cleaned.replace(",", " "), sep=" ")
            return values.astype(float).tolist() if values.size else None
    if isinstance(parsed, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in parsed]
        except (TypeError, ValueError):
            return None
    return None


def load_targets() -> pd.DataFrame:
    """Load the canonical, leak-free 70/15/15 dataset used by other models."""
    train, valid, test = build_dataset(required_cols=JISDOR_FEATURES)
    target = pd.concat([train, valid, test], ignore_index=True)
    target["date"] = pd.to_datetime(target["date"]).dt.normalize()
    target["volatility_class"] = target["volatility_class"].astype(str)
    return target[["date", "volatility_class", "realized_vol", *JISDOR_FEATURES]]


def normalize_date(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, str]:
    candidates = ("date", "target_date", "published_date", "datetime", "timestamp")
    lookup = {str(col).strip().lower(): col for col in df.columns}
    date_col = next((lookup[c] for c in candidates if c in lookup), None)
    if date_col is None:
        raise ValueError(f"Kolom tanggal tidak ditemukan pada {path}. Kolom: {list(df.columns)}")
    result = df.copy()
    result["date"] = pd.to_datetime(result[date_col], errors="coerce").dt.normalize()
    result = result.dropna(subset=["date"])
    return result, date_col


def load_embedding(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File embedding tidak ditemukan: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"File embedding kosong: {path}")
    df, _ = normalize_date(raw, path)

    vector_names = {
        "embedding", "embeddings", "vector", "features", "pooled_embedding",
        "mean_embedding", "sentence_embedding", "document_embedding",
    }
    vector_cols = [c for c in df.columns if str(c).strip().lower() in vector_names]

    if vector_cols:
        vectors = [parse_vector(v) for v in df[vector_cols[0]]]
        lengths = [len(v) for v in vectors if v is not None]
        if not lengths:
            raise ValueError(f"Tidak ada embedding vector yang valid pada {path}")
        dimensions = max(set(lengths), key=lengths.count)
        matrix = np.zeros((len(df), dimensions), dtype=np.float32)
        for i, vector in enumerate(vectors):
            if vector is not None and len(vector) == dimensions:
                matrix[i] = np.asarray(vector, dtype=np.float32)
        features = pd.DataFrame(matrix, columns=[f"{prefix}_dim_{i}" for i in range(dimensions)])
        features.insert(0, "date", df["date"].to_numpy())
    else:
        explicit_pattern = re.compile(
            r"^(embedding|emb|feature|dim|vector|hidden|pool|cls|mean_pool|finbert|roberta|qwen)[_\-.]?\d+$",
            flags=re.IGNORECASE,
        )
        cols = [c for c in df.columns if explicit_pattern.match(str(c).strip())]
        if not cols:
            cols = [c for c in df.columns if re.fullmatch(r"\d+", str(c).strip())]
        if not cols:
            excluded = {
                "date", "target_date", "published_date", "datetime", "timestamp",
                "direction", "volatility_class", "log_return", "realized_vol", "return",
                "delta", "kurs", "prev_kurs", "prev_date", "gap_days", "n_news",
                "article_count", "n_tokens", "mean_tokens", "relevance_score", "section",
                "category", "text", "text_norm", "url", "model", "target_date_lag1",
            }
            cols = [
                c for c in df.columns
                if str(c).strip().lower() not in excluded
                and pd.api.types.is_numeric_dtype(df[c])
            ]
        if not cols:
            raise ValueError(f"Tidak menemukan kolom embedding numerik pada {path}")
        features = df[["date", *cols]].copy()
        features = features.rename(columns={c: f"{prefix}_{i}" for i, c in enumerate(cols)})

    feature_cols = [c for c in features.columns if c != "date"]
    for col in feature_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.groupby("date", as_index=False)[feature_cols].mean()
    return features


def load_daily_numeric_features(path: Path, prefix: str, one_hot_dominant_topic: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File fitur tidak ditemukan: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"File fitur kosong: {path}")
    df, _ = normalize_date(raw, path)

    parts: list[pd.DataFrame] = [df[["date"]].reset_index(drop=True)]
    excluded = {
        "date", "target_date", "published_date", "datetime", "timestamp",
        "model", "dominant_topic" if one_hot_dominant_topic else "__none__",
        "url", "document_id", "entity_text", "normalized_entity", "entity_label",
        "context", "section", "text", "text_norm",
    }

    numeric_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if numeric_cols:
        numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.rename(columns={c: f"{prefix}_{c}" for c in numeric_cols})
        parts.append(numeric.reset_index(drop=True))

    if one_hot_dominant_topic and "dominant_topic" in df.columns:
        dominant = pd.to_numeric(df["dominant_topic"], errors="coerce").fillna(-1).astype(int).astype(str)
        dummies = pd.get_dummies(dominant, prefix=f"{prefix}_dominant_topic", dtype=float)
        parts.append(dummies.reset_index(drop=True))

    features = pd.concat(parts, axis=1)
    feature_cols = [c for c in features.columns if c != "date"]
    if not feature_cols:
        raise ValueError(f"Tidak ada fitur numerik yang dapat digunakan dari {path}")

    features = features.replace([np.inf, -np.inf], np.nan)
    # Daily files are expected to be one row/date; average duplicate numeric rows.
    features = features.groupby("date", as_index=False)[feature_cols].mean()
    return features


def make_classifiers() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=LOGISTIC_C, max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("classifier", XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=300, max_depth=3,
                                         learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                         eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def evaluate_one_configuration(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    *,
    model: str,
    text_type: str,
    experiment: str,
    feature_coverage: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    n_train = int(len(X) * TRAIN_FRAC)
    n_valid = int(len(X) * VALID_FRAC)
    train_idx = np.arange(n_train)
    split_indices = {
        "validation": np.arange(n_train, n_train + n_valid),
        "test": np.arange(n_train + n_valid, len(X)),
    }
    y_train = y[train_idx]
    if np.unique(y_train).size < 2:
        raise ValueError(f"{model}/{text_type}/{experiment}: train tidak memiliki minimal dua kelas.")

    class_to_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    y_train_encoded = np.asarray([class_to_index[label] for label in y_train], dtype=int)
    for classifier_name, clf_template in make_classifiers().items():
        clf_template.fit(X[train_idx], y_train_encoded)
        for evaluation_split, eval_idx in split_indices.items():
            y_eval = y[eval_idx]
            pred_encoded = clf_template.predict(X[eval_idx]).astype(int)
            pred = np.asarray([CLASS_ORDER[index] for index in pred_encoded])
            report = classification_report(y_eval, pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
            row = {
                "model": model, "text_type": text_type, "experiment": experiment, "classifier": classifier_name,
                "evaluation_split": evaluation_split, "n_train": int(len(train_idx)), "n_validation": int(len(split_indices["validation"])), "n_test": int(len(split_indices["test"])),
                "train_start": str(dates.iloc[train_idx].min().date()), "train_end": str(dates.iloc[train_idx].max().date()),
                "evaluation_start": str(dates.iloc[eval_idx].min().date()), "evaluation_end": str(dates.iloc[eval_idx].max().date()),
                "n_features": int(X.shape[1]), "accuracy": float(accuracy_score(y_eval, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_eval, pred)),
                "macro_precision": float(precision_score(y_eval, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_eval, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(y_eval, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_eval, pred, labels=CLASS_ORDER, average="weighted", zero_division=0)),
                "f1_low": float(report["low"]["f1-score"]), "f1_medium": float(report["medium"]["f1-score"]),
                "f1_high": float(report["high"]["f1-score"]), **feature_coverage,
            }
            fold_rows.append(row)
            for label in CLASS_ORDER:
                class_rows.append({
                    "model": model, "text_type": text_type, "experiment": experiment, "classifier": classifier_name,
                    "evaluation_split": evaluation_split, "class": label, "precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]), "f1_score": float(report[label]["f1-score"]),
                    "support": int(report[label]["support"]),
                })

    return fold_rows, class_rows


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    metric_cols = [
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "f1_low", "f1_medium", "f1_high",
    ]
    group_cols = ["model", "text_type", "experiment", "classifier", "evaluation_split"]
    rows = []
    for keys, group in folds.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_evaluations"] = int(len(group))
        row["n_features"] = int(group["n_features"].iloc[0])
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        for col in ["embedding_coverage_pct", "topic_coverage_pct", "ner_coverage_pct", "temporal_coverage_pct"]:
            if col in group.columns:
                row[col] = float(group[col].iloc[0])
        rows.append(row)
    result = pd.DataFrame(rows)
    baseline = result[result["experiment"] == "E0_jisdor_price_only"][["classifier", "evaluation_split", "macro_f1_mean"]].rename(
        columns={"macro_f1_mean": "jisdor_price_only_macro_f1_mean"}
    )
    result = result.merge(baseline, on=["classifier", "evaluation_split"], how="left")
    result["delta_macro_f1_vs_jisdor_price_only"] = result["macro_f1_mean"] - result["jisdor_price_only_macro_f1_mean"]
    return result.sort_values(["macro_f1_mean", "balanced_accuracy_mean"], ascending=False).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Project root: %s", ROOT_DIR)
    logger.info("Topic model: %s", TOPIC_MODEL)
    logger.info("Output: %s", OUTPUT_DIR)

    targets = load_targets()
    if len(targets) < 20:
        raise ValueError(f"Jumlah target ({len(targets)}) tidak cukup untuk split 70/15/15.")
    logger.info("Target rows: %s | classes: %s", len(targets), targets["volatility_class"].value_counts().to_dict())

    # Topic and NER are loaded once and reused for each embedding configuration.
    topic_features = load_daily_numeric_features(TOPIC_FILE, "topic", one_hot_dominant_topic=True)
    ner_features = load_daily_numeric_features(NER_FILE, "ner", one_hot_dominant_topic=False)
    temporal_features = load_daily_numeric_features(TEMPORAL_FILE, "temporal", one_hot_dominant_topic=False)

    all_fold_rows: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    # Baseline required to establish whether text fusion adds anything beyond
    # the market state available before the prediction day.
    price_matrix = targets[JISDOR_FEATURES].to_numpy(dtype=np.float32)
    baseline_rows, baseline_class_rows = evaluate_one_configuration(
        price_matrix, targets["volatility_class"].to_numpy(), targets["date"].reset_index(drop=True),
        model="jisdor", text_type="price_only", experiment="E0_jisdor_price_only",
        feature_coverage={"embedding_coverage_pct": 0.0, "topic_coverage_pct": 0.0, "ner_coverage_pct": 0.0},
    )
    all_fold_rows.extend(baseline_rows)
    all_class_rows.extend(baseline_class_rows)

    for model in MODELS:
        for text_type in TEXT_TYPES:
            embedding_path = EMBEDDING_DIR / text_type / f"{model}_{text_type}_features.csv"
            if not embedding_path.exists():
                skipped.append({"model": model, "text_type": text_type, "reason": f"Missing embedding: {embedding_path}"})
                logger.warning("Skip %s/%s: file embedding tidak ditemukan", model, text_type)
                continue

            logger.info("\nLoading %s / %s", model, text_type)
            embedding_features = load_embedding(embedding_path, "embedding")

            # Merge every source onto the exact same chronological target dates.
            base = targets[["date", "volatility_class", "realized_vol", *JISDOR_FEATURES]].copy()
            price_cols = JISDOR_FEATURES
            merged_embedding = base.merge(embedding_features, on="date", how="left", validate="one_to_one")
            emb_cols = [c for c in embedding_features.columns if c != "date"]
            emb_missing = merged_embedding[emb_cols].isna().all(axis=1)
            X_embedding = merged_embedding[price_cols + emb_cols].fillna(0.0).to_numpy(dtype=np.float32)

            merged_topic = base.merge(embedding_features, on="date", how="left", validate="one_to_one")
            merged_topic = merged_topic.merge(topic_features, on="date", how="left", validate="one_to_one")
            topic_cols = [c for c in topic_features.columns if c != "date"]
            topic_missing = merged_topic[topic_cols].isna().all(axis=1)
            X_embedding_topic = merged_topic[price_cols + emb_cols + topic_cols].fillna(0.0).to_numpy(dtype=np.float32)

            merged_ner = base.merge(embedding_features, on="date", how="left", validate="one_to_one")
            merged_ner = merged_ner.merge(ner_features, on="date", how="left", validate="one_to_one")
            ner_cols = [c for c in ner_features.columns if c != "date"]
            ner_missing = merged_ner[ner_cols].isna().all(axis=1)
            X_embedding_ner = merged_ner[price_cols + emb_cols + ner_cols].fillna(0.0).to_numpy(dtype=np.float32)

            merged_all = base.merge(embedding_features, on="date", how="left", validate="one_to_one")
            merged_all = merged_all.merge(topic_features, on="date", how="left", validate="one_to_one")
            merged_all = merged_all.merge(ner_features, on="date", how="left", validate="one_to_one")
            X_all = merged_all[price_cols + emb_cols + topic_cols + ner_cols].fillna(0.0).to_numpy(dtype=np.float32)

            temporal_cols = [c for c in temporal_features.columns if c != "date"]
            merged_temporal = base.merge(embedding_features, on="date", how="left", validate="one_to_one")
            merged_temporal = merged_temporal.merge(temporal_features, on="date", how="left", validate="one_to_one")
            temporal_missing = merged_temporal[temporal_cols].isna().all(axis=1)
            X_embedding_temporal = merged_temporal[price_cols + emb_cols + temporal_cols].fillna(0.0).to_numpy(dtype=np.float32)

            merged_topic_temporal = merged_temporal.merge(topic_features, on="date", how="left", validate="one_to_one")
            X_embedding_topic_temporal = merged_topic_temporal[price_cols + emb_cols + topic_cols + temporal_cols].fillna(0.0).to_numpy(dtype=np.float32)

            merged_ner_temporal = merged_temporal.merge(ner_features, on="date", how="left", validate="one_to_one")
            X_embedding_ner_temporal = merged_ner_temporal[price_cols + emb_cols + ner_cols + temporal_cols].fillna(0.0).to_numpy(dtype=np.float32)

            merged_all_temporal = merged_all.merge(temporal_features, on="date", how="left", validate="one_to_one")
            X_all_temporal = merged_all_temporal[price_cols + emb_cols + topic_cols + ner_cols + temporal_cols].fillna(0.0).to_numpy(dtype=np.float32)

            y = base["volatility_class"].to_numpy()
            dates = base["date"].reset_index(drop=True)
            coverage = {
                "embedding_coverage_pct": float(100 * (~emb_missing).mean()),
                "topic_coverage_pct": float(100 * (~topic_missing).mean()),
                "ner_coverage_pct": float(100 * (~ner_missing).mean()),
                "temporal_coverage_pct": float(100 * (~temporal_missing).mean()),
            }
            configurations = {
                "E1_jisdor_plus_embedding": X_embedding,
                "E2_jisdor_plus_embedding_topic": X_embedding_topic,
                "E3_jisdor_plus_embedding_ner": X_embedding_ner,
                "E4_jisdor_plus_embedding_topic_ner": X_all,
                "E5_jisdor_plus_embedding_temporal": X_embedding_temporal,
                "E6_jisdor_plus_embedding_topic_temporal": X_embedding_topic_temporal,
                "E7_jisdor_plus_embedding_ner_temporal": X_embedding_ner_temporal,
                "E8_jisdor_plus_embedding_topic_ner_temporal": X_all_temporal,
            }

            for experiment, matrix in configurations.items():
                logger.info("Evaluate %s | %s | %s | shape=%s", model, text_type, experiment, matrix.shape)
                fold_rows, class_rows = evaluate_one_configuration(
                    matrix, y, dates,
                    model=model,
                    text_type=text_type,
                    experiment=experiment,
                    feature_coverage=coverage,
                )
                all_fold_rows.extend(fold_rows)
                all_class_rows.extend(class_rows)

    if not all_fold_rows:
        raise RuntimeError("Tidak ada eksperimen yang berhasil. Periksa lokasi file embedding, topic, dan NER.")

    fold_df = pd.DataFrame(all_fold_rows)
    class_df = pd.DataFrame(all_class_rows)
    summary_df = summarize(fold_df)

    fold_path = OUTPUT_DIR / "feature_fusion_fold_metrics.csv"
    summary_path = OUTPUT_DIR / "feature_fusion_summary.csv"
    class_path = OUTPUT_DIR / "feature_fusion_class_metrics.csv"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    class_df.to_csv(class_path, index=False)

    metadata = {
        "target_file": str(TARGET_FILE),
        "target": "volatility_class (low/medium/high)",
        "target_definition": "5-BI-business-day rolling standard deviation of log_return; tertile thresholds fitted on earliest 70% chronological observations, consistent with project target.",
        "topic_model": TOPIC_MODEL,
        "topic_file": str(TOPIC_FILE),
        "ner_file": str(NER_FILE),
        "temporal_file": str(TEMPORAL_FILE),
        "embedding_dir": str(EMBEDDING_DIR),
        "experiments": {name: list(features) for name, features in EXPERIMENTS.items()},
        "models": list(MODELS),
        "text_types": list(TEXT_TYPES),
        "split_strategy": "chronological fixed split: 70% train, 15% validation, 15% test",
        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VALID_FRAC,
        "test_fraction": 1 - TRAIN_FRAC - VALID_FRAC,
        "classifiers": ["StandardScaler + LogisticRegression(class_weight=balanced, L2)", "SimpleImputer + XGBoost multiclass"],
        "jisdor_feature_columns": JISDOR_FEATURES,
        "n_target_rows": int(len(targets)),
        "class_distribution": {k: int(v) for k, v in targets["volatility_class"].value_counts().items()},
        "skipped_embedding_files": skipped,
        "caveat": "Topic models were generated before this supervised evaluation. If topic models were fitted on the full time span, topic feature extraction is transductive; for strict forecasting claims, fit topic models using only the 70% training period before transforming validation/test.",
    }
    with (OUTPUT_DIR / "feature_fusion_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("\nFEATURE FUSION EVALUATION FINISHED")
    logger.info("Fold metrics : %s", fold_path)
    logger.info("Summary      : %s", summary_path)
    logger.info("Class metrics: %s", class_path)
    logger.info("Metadata     : %s", OUTPUT_DIR / "feature_fusion_metadata.json")
    print("\nTop configurations by macro-F1:")
    print(summary_df[["model", "text_type", "classifier", "experiment", "macro_f1_mean", "balanced_accuracy_mean", "delta_macro_f1_vs_jisdor_price_only"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
