
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "headlineVsFulltext"

ALIGNED_PATH = DATA_DIR / "aligned_daily.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Three-class classification
CLASS_MAPPING = {
    "down": 0,
    "up": 1,
    "flat": 2,
}

CLASS_NAMES = ["down", "up", "flat"]
CLASS_LABELS = [0, 1, 2]

EXPERIMENTS = {
    "word2vec_headline": "word2vec_headline_features.csv",
    "fasttext_headline": "fasttext_headline_features.csv",
    "doc2vec_headline": "doc2vec_headline_features.csv",
    "finbert_headline": "finbert_headline_features.csv",
    "qwen_headline": "qwen_headline_features.csv",
    "roberta_headline": "roberta_headline_features.csv",

    "word2vec_fulltext": "word2vec_fulltext_features.csv",
    "fasttext_fulltext": "fasttext_fulltext_features.csv",
    "doc2vec_fulltext": "doc2vec_fulltext_features.csv",
    "finbert_fulltext": "finbert_fulltext_features.csv",
    "qwen_fulltext": "qwen_fulltext_features.csv",
    "roberta_fulltext": "roberta_fulltext_features.csv",
}


# ============================================================
# 2. TARGET PREPROCESSING
# ============================================================

def normalize_target(series):
    """
    Three-class classification:
        down / turun = 0
        up / naik    = 1
        flat / datar = 2
    """

    text = series.astype(str).str.strip().str.lower()

    mapping = {
        "down": 0,
        "downward": 0,
        "turun": 0,
        "negative": 0,

        "up": 1,
        "upward": 1,
        "naik": 1,
        "positive": 1,

        "flat": 2,
        "datar": 2,
        "neutral": 2,
    }

    result = text.map(mapping)

    if result.isna().any():
        unknown = text[result.isna()].unique()
        raise ValueError(
            f"Nilai direction tidak dikenali: {unknown}"
        )

    return result.astype(int)


# ============================================================
# 3. SELECT EMBEDDING FEATURES
# ============================================================

def get_feature_columns(df):

    excluded_keywords = [
        "count",
        "coverage",
        "token_count",
        "headline_count",
        "n_tokens",
    ]

    feature_columns = []

    for col in df.columns:

        if col == "date":
            continue

        col_lower = col.lower()

        if any(
            keyword in col_lower
            for keyword in excluded_keywords
        ):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)

    if not feature_columns:
        raise ValueError(
            "Tidak ditemukan kolom embedding numerik."
        )

    return feature_columns


# ============================================================
# 4. EVALUATION METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    y_true_bin = label_binarize(
        y_true,
        classes=CLASS_LABELS,
    )

    # Multiclass ROC-AUC (One-vs-Rest, macro)
    roc_auc = roc_auc_score(
        y_true,
        y_prob,
        labels=CLASS_LABELS,
        multi_class="ovr",
        average="macro",
    )

    # Macro PR-AUC: average precision per class
    pr_auc = average_precision_score(
        y_true_bin,
        y_prob,
        average="macro",
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),

        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# 5. LOAD TARGET DATA
# ============================================================

print("=" * 65)
print("LOADING TARGET DATA")
print("=" * 65)

if not ALIGNED_PATH.exists():
    raise FileNotFoundError(
        f"Target file tidak ditemukan: {ALIGNED_PATH}"
    )

aligned = pd.read_csv(ALIGNED_PATH)

required_columns = ["date", "direction"]

for col in required_columns:
    if col not in aligned.columns:
        raise ValueError(
            f"Kolom '{col}' tidak ditemukan di aligned_daily.csv"
        )

aligned["date"] = pd.to_datetime(
    aligned["date"],
    errors="coerce",
)

aligned = aligned.dropna(
    subset=["date", "direction"]
).copy()

# Keep all three classes
aligned["direction"] = normalize_target(
    aligned["direction"]
)

aligned = aligned.sort_values(
    "date"
).reset_index(drop=True)

if aligned["date"].duplicated().any():
    raise ValueError(
        "Terdapat tanggal duplikat di aligned_daily.csv."
    )

print(f"Total observations: {len(aligned)}")

print("\nClass distribution:")
print(
    aligned["direction"]
    .map({
        0: "Down",
        1: "Up",
        2: "Flat",
    })
    .value_counts()
)


# ============================================================
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

split_index = int(
    len(aligned) * (1 - TEST_SIZE)
)

train_dates = aligned.iloc[
    :split_index
]["date"]

test_dates = aligned.iloc[
    split_index:
]["date"]

print("\n" + "=" * 65)
print("CHRONOLOGICAL TRAIN-TEST SPLIT")
print("=" * 65)

print(f"Train observations: {len(train_dates)}")
print(f"Test observations : {len(test_dates)}")

print(
    f"Train period: {train_dates.min().date()} "
    f"to {train_dates.max().date()}"
)

print(
    f"Test period : {test_dates.min().date()} "
    f"to {test_dates.max().date()}"
)


# ============================================================
# 7. TRAIN AND EVALUATE ALL EXPERIMENTS
# ============================================================

all_results = []
all_predictions = []

for experiment_name, filename in EXPERIMENTS.items():

    feature_path = DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 65)

    if not feature_path.exists():
        print(f"[SKIPPED] File tidak ditemukan: {feature_path}")
        continue

    # --------------------------------------------------------
    # LOAD EMBEDDING FEATURES
    # --------------------------------------------------------

    features = pd.read_csv(feature_path)

    if "date" not in features.columns:
        print("[SKIPPED] Kolom date tidak ditemukan.")
        continue

    features["date"] = pd.to_datetime(
        features["date"],
        errors="coerce",
    )

    features = features.dropna(
        subset=["date"]
    ).copy()

    if features["date"].duplicated().any():
        raise ValueError(
            f"{experiment_name}: ditemukan date duplikat."
        )

    feature_columns = get_feature_columns(features)

    print(
        f"Embedding dimensions: {len(feature_columns)}"
    )

    features[feature_columns] = features[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # --------------------------------------------------------
    # MERGE TARGET AND EMBEDDING
    # --------------------------------------------------------

    dataset = aligned[
        ["date", "direction"]
    ].merge(
        features[["date"] + feature_columns],
        on="date",
        how="left",
        validate="one_to_one",
    )

    dataset = dataset.sort_values(
        "date"
    ).reset_index(drop=True)

    dataset[feature_columns] = (
        dataset[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    missing_embedding_dates = (
        dataset[feature_columns]
        .eq(0)
        .all(axis=1)
        .sum()
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    train_data = dataset[
        dataset["date"].isin(train_dates)
    ].copy()

    test_data = dataset[
        dataset["date"].isin(test_dates)
    ].copy()

    if len(train_data) == 0 or len(test_data) == 0:
        print("[SKIPPED] Data train/test kosong.")
        continue

    X_train = train_data[feature_columns]
    y_train = train_data["direction"]

    X_test = test_data[feature_columns]
    y_test = test_data["direction"]

    if set(CLASS_LABELS) - set(y_train.unique()):
        print(
            "[SKIPPED] Training tidak memiliki seluruh "
            "tiga kelas target."
        )
        continue

    if set(CLASS_LABELS) - set(y_test.unique()):
        print(
            "[SKIPPED] Testing tidak memiliki seluruh "
            "tiga kelas target; multiclass AUC tidak dapat "
            "dibandingkan secara konsisten."
        )
        continue

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")
    print(
        f"Dates with all-zero embeddings: "
        f"{missing_embedding_dates}"
    )

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value=0,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)

    classifier = model.named_steps["classifier"]

    # Ensure probability columns follow class order [0, 1, 2]
    class_order = list(classifier.classes_)

    y_prob = y_prob[
        :,
        [class_order.index(label) for label in CLASS_LABELS]
    ]

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob,
    )

    result = {
        "experiment": experiment_name,
        "model": experiment_name.rsplit("_", 1)[0],
        "text_type": experiment_name.rsplit("_", 1)[1],
        "n_features": len(feature_columns),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "down_test": int((y_test == 0).sum()),
        "up_test": int((y_test == 1).sum()),
        "flat_test": int((y_test == 2).sum()),
        "zero_embedding_dates": int(missing_embedding_dates),
        **metrics,
    }

    all_results.append(result)

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    predictions = pd.DataFrame({
        "date": test_data["date"].values,
        "experiment": experiment_name,
        "y_true": y_test.values,
        "y_pred": y_pred,
        "prob_down": y_prob[:, 0],
        "prob_up": y_prob[:, 1],
        "prob_flat": y_prob[:, 2],
    })

    all_predictions.append(predictions)

    # --------------------------------------------------------
    # PRINT METRICS
    # --------------------------------------------------------

    print("\nEvaluation results:")

    for metric, value in metrics.items():
        print(f"{metric:18s}: {value:.4f}")


# ============================================================
# 8. SAVE RESULTS
# ============================================================

if not all_results:
    raise RuntimeError(
        "Tidak ada eksperimen yang berhasil. "
        "Periksa file embedding dan distribusi kelas."
    )

results_df = pd.DataFrame(
    all_results
)

results_path = (
    OUTPUT_DIR / "experiment_results.csv"
)

results_df.to_csv(
    results_path,
    index=False,
)

if all_predictions:

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_path = (
        OUTPUT_DIR / "test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================
print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Down", "Up", "Flat"],
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )
)


print("\n" + "=" * 65)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 65)

print(
    results_df[
        [
            "experiment",
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nSaved files:")
print(results_path)

if all_predictions:
    print(predictions_path)
