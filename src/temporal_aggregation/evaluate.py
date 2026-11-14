# ============================================================
# evaluate_temporal_aggregation.py
#
# Evaluasi A0-A4 Temporal Aggregation
# Target       : direction (down / flat / up)
# Validation   : Chronological Walk-Forward CV
# Classifier   : PyTorch Linear Classifier
# Device       : Apple MPS -> CUDA -> CPU
#
# IMPORTANT:
# - Feature scaling hanya fit pada TRAIN fold
# - Tidak ada random train/test split
# - Classifier training menggunakan PyTorch
# - MPS digunakan jika tersedia
# ============================================================

from pathlib import Path
import json
import random
import time
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ALIGNED_DAILY_CSV = ROOT / "data/processed/aligned_daily.csv"

TEMPORAL_AGG_DIR = ROOT / "data/processed/temporal_aggregation"

OUTPUT_DIR = ROOT / "data/processed/temporal_evaluation"

RESULTS_DIR = OUTPUT_DIR / "results"

SUMMARY_CSV = OUTPUT_DIR / "temporal_aggregation_evaluation.csv"
RANKING_CSV = OUTPUT_DIR / "temporal_aggregation_ranking.csv"


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

TARGET_COLUMN = "direction"

TARGET_CLASSES = [
    "down",
    "flat",
    "up",
]

CLASS_TO_IDX = {
    label: idx
    for idx, label in enumerate(TARGET_CLASSES)
}


# ------------------------------------------------------------
# Walk-forward CV
# ------------------------------------------------------------

N_FOLDS = 4

# Minimum number of training samples
MIN_TRAIN_SAMPLES = 100

# Test size per fold
TEST_SIZE = 200


# ------------------------------------------------------------
# PyTorch training
# ------------------------------------------------------------

RANDOM_SEED = 42

EPOCHS = 150

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

BATCH_SIZE = 128

PATIENCE = 15

MIN_DELTA = 1e-5


# ------------------------------------------------------------
# Numerical stability
# ------------------------------------------------------------

STD_EPS = 1e-8

MAX_FEATURE_VALUE = 1e6


# ------------------------------------------------------------
# Resume
# ------------------------------------------------------------

# True:
#   skip configuration if result CSV already exists
#
# False:
#   rerun everything
RESUME = True


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(f"[INFO] {message}", flush=True)


def log_warning(message):
    print(f"[WARNING] {message}", flush=True)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    # CUDA first
    if torch.cuda.is_available():

        device = torch.device("cuda")

        log("GPU detected: CUDA")
        log(f"Device: {torch.cuda.get_device_name(0)}")

        return device

    # Apple Silicon
    if hasattr(torch.backends, "mps"):

        if (
            torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):

            device = torch.device("mps")

            log("GPU detected: Apple MPS")
            log("Device: mps")

            return device

    # CPU fallback
    device = torch.device("cpu")

    log("GPU tidak tersedia. Menggunakan CPU.")

    return device


# ============================================================
# DIRECTORY
# ============================================================

def ensure_dirs():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD CALENDAR
# ============================================================

def load_calendar():

    if not ALIGNED_DAILY_CSV.exists():

        raise FileNotFoundError(
            f"Calendar tidak ditemukan:\n{ALIGNED_DAILY_CSV}"
        )

    df = pd.read_csv(ALIGNED_DAILY_CSV)

    required = {
        "date",
        TARGET_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"Kolom calendar hilang: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():

        raise ValueError(
            "Terdapat date yang tidak valid pada aligned_daily.csv"
        )

    df = df.sort_values("date").reset_index(drop=True)

    invalid_labels = set(
        df[TARGET_COLUMN].dropna().unique()
    ) - set(TARGET_CLASSES)

    if invalid_labels:

        raise ValueError(
            f"Label direction tidak dikenal: "
            f"{sorted(invalid_labels)}"
        )

    return df


# ============================================================
# DISCOVER AGGREGATION FILES
# ============================================================

def discover_aggregation_files():

    files = []

    if not TEMPORAL_AGG_DIR.exists():

        raise FileNotFoundError(
            f"Folder temporal aggregation tidak ditemukan:\n"
            f"{TEMPORAL_AGG_DIR}"
        )

    for model_dir in sorted(TEMPORAL_AGG_DIR.iterdir()):

        if not model_dir.is_dir():
            continue

        for csv_path in sorted(
            model_dir.glob("*.csv")
        ):

            name = csv_path.name

            # Skip diagnostics
            if "diagnostics" in name.lower():
                continue

            # Only A0-A4
            if not any(
                f"_A{i}_"
                in name
                for i in range(5)
            ):
                continue

            files.append(
                {
                    "model": model_dir.name,
                    "file": csv_path,
                    "filename": name,
                }
            )

    return files


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

def find_feature_columns(df):

    excluded = {
        "date",
        TARGET_COLUMN,
        "direction",
        "news_count",
        "attention_news_count",
    }

    feature_cols = []

    for col in df.columns:

        if col in excluded:
            continue

        # Numeric columns only
        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            feature_cols.append(col)

    if not feature_cols:

        raise ValueError(
            "Tidak ditemukan feature numerik."
        )

    return feature_cols


# ============================================================
# LOAD AGGREGATION DATA
# ============================================================

def load_aggregation_data(
    file_info,
    calendar
):

    path = file_info["file"]

    df = pd.read_csv(path)

    if "date" not in df.columns:

        raise ValueError(
            f"{path.name}: tidak memiliki kolom date"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.merge(
        calendar[
            ["date", TARGET_COLUMN]
        ],
        on="date",
        how="inner",
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    feature_cols = find_feature_columns(df)

    X = (
        df[feature_cols]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .astype(np.float32)
    )

    # Missing aggregation features
    # are treated as zero.
    X = X.fillna(0.0)

    X = X.clip(
        -MAX_FEATURE_VALUE,
        MAX_FEATURE_VALUE
    )

    y = df[TARGET_COLUMN].map(
        CLASS_TO_IDX
    )

    valid = y.notna()

    df = df.loc[valid].reset_index(drop=True)

    X = X.loc[valid].reset_index(drop=True)

    y = y.loc[valid].astype(int).to_numpy()

    return (
        df,
        X.to_numpy(dtype=np.float32),
        y,
        feature_cols,
    )


# ============================================================
# STANDARDIZATION
# ============================================================

def fit_scaler(X_train):

    mean = np.mean(
        X_train,
        axis=0,
        dtype=np.float64
    ).astype(np.float32)

    std = np.std(
        X_train,
        axis=0,
        dtype=np.float64
    ).astype(np.float32)

    # Avoid division by zero
    std[std < STD_EPS] = 1.0

    # Safety
    mean = np.nan_to_num(
        mean,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    std = np.nan_to_num(
        std,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    std[std < STD_EPS] = 1.0

    return mean, std


def transform_features(
    X,
    mean,
    std
):

    X_scaled = (
        X - mean
    ) / std

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_scaled = np.clip(
        X_scaled,
        -20.0,
        20.0
    )

    return X_scaled.astype(
        np.float32
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(
    y_train,
    device
):

    counts = np.bincount(
        y_train,
        minlength=len(TARGET_CLASSES)
    ).astype(np.float32)

    weights = np.ones(
        len(TARGET_CLASSES),
        dtype=np.float32
    )

    total = counts.sum()

    for i in range(
        len(TARGET_CLASSES)
    ):

        if counts[i] > 0:

            weights[i] = (
                total
                /
                (
                    len(TARGET_CLASSES)
                    * counts[i]
                )
            )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )


# ============================================================
# PYTORCH CLASSIFIER
# ============================================================

class LinearClassifier(nn.Module):

    """
    Multinomial logistic regression equivalent.

    logits = XW + b

    Softmax is handled internally by
    CrossEntropyLoss.
    """

    def __init__(
        self,
        input_dim,
        n_classes
    ):

        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            n_classes
        )

    def forward(self, x):

        return self.linear(x)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
):

    input_dim = X_train.shape[1]

    model = LinearClassifier(
        input_dim=input_dim,
        n_classes=len(TARGET_CLASSES)
    ).to(device)

    class_weights = compute_class_weights(
        y_train,
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Convert data
    # --------------------------------------------------------

    X_train_tensor = torch.from_numpy(
        X_train
    ).to(device)

    y_train_tensor = torch.from_numpy(
        y_train.astype(np.int64)
    ).to(device)

    X_val_tensor = torch.from_numpy(
        X_val
    ).to(device)

    y_val_tensor = torch.from_numpy(
        y_val.astype(np.int64)
    ).to(device)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=min(
            BATCH_SIZE,
            len(dataset)
        ),
        shuffle=True,
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_state = None

    patience_counter = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        epoch_loss = 0.0

        for batch_x, batch_y in loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.detach().item()
                * len(batch_x)
            )

        epoch_loss /= len(dataset)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val_tensor
            )

            val_loss = criterion(
                val_logits,
                y_val_tensor
            ).item()

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if (
            best_val_loss
            - val_loss
            > MIN_DELTA
        ):

            best_val_loss = val_loss

            best_state = {
                key: value.detach().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"        Epoch "
                f"{epoch:03d}/{EPOCHS} "
                f"| train_loss={epoch_loss:.5f} "
                f"| val_loss={val_loss:.5f}",
                flush=True
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= PATIENCE:

            print(
                f"        Early stopping "
                f"at epoch {epoch}",
                flush=True
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    X,
    device
):

    model.eval()

    X_tensor = torch.from_numpy(
        X.astype(np.float32)
    ).to(device)

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

    return (
        predictions
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
            2
        ]
    )

    metrics = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "macro_precision":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "mcc":
            matthews_corrcoef(
                y_true,
                y_pred
            ),

        "f1_down":
            f1_score(
                y_true,
                y_pred,
                labels=[0],
                average="macro",
                zero_division=0,
            ),

        "f1_flat":
            f1_score(
                y_true,
                y_pred,
                labels=[1],
                average="macro",
                zero_division=0,
            ),

        "f1_up":
            f1_score(
                y_true,
                y_pred,
                labels=[2],
                average="macro",
                zero_division=0,
            ),

        "precision_down":
            precision_score(
                y_true,
                y_pred,
                labels=[0],
                average="macro",
                zero_division=0,
            ),

        "precision_flat":
            precision_score(
                y_true,
                y_pred,
                labels=[1],
                average="macro",
                zero_division=0,
            ),

        "precision_up":
            precision_score(
                y_true,
                y_pred,
                labels=[2],
                average="macro",
                zero_division=0,
            ),

        "recall_down":
            recall_score(
                y_true,
                y_pred,
                labels=[0],
                average="macro",
                zero_division=0,
            ),

        "recall_flat":
            recall_score(
                y_true,
                y_pred,
                labels=[1],
                average="macro",
                zero_division=0,
            ),

        "recall_up":
            recall_score(
                y_true,
                y_pred,
                labels=[2],
                average="macro",
                zero_division=0,
            ),
    }

    return metrics, cm


# ============================================================
# CREATE WALK-FORWARD FOLDS
# ============================================================

def create_folds(
    n_samples
):

    folds = []

    train_end = MIN_TRAIN_SAMPLES

    while (
        train_end < n_samples
        and len(folds) < N_FOLDS
    ):

        test_start = train_end

        test_end = min(
            test_start + TEST_SIZE,
            n_samples
        )

        if test_start >= test_end:
            break

        train_idx = np.arange(
            0,
            train_end
        )

        test_idx = np.arange(
            test_start,
            test_end
        )

        folds.append(
            {
                "fold":
                    len(folds) + 1,

                "train_idx":
                    train_idx,

                "test_idx":
                    test_idx,
            }
        )

        train_end = test_end

    return folds


# ============================================================
# EVALUATE ONE AGGREGATION FILE
# ============================================================

def evaluate_one_file(
    file_info,
    calendar,
    device,
):

    model_name = file_info["model"]

    filename = file_info["filename"]

    path = file_info["file"]

    print()
    print("=" * 70)
    print(f"Model       : {model_name}")
    print(f"Aggregation : {filename}")
    print(f"File        : {filename}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df, X, y, feature_cols = load_aggregation_data(
        file_info,
        calendar
    )

    print(
        f"Samples     : {len(X):,}"
    )

    print(
        f"Features    : {X.shape[1]:,}"
    )

    # --------------------------------------------------------
    # Folds
    # --------------------------------------------------------

    folds = create_folds(
        len(X)
    )

    if not folds:

        raise ValueError(
            "Tidak cukup data untuk membuat fold."
        )

    all_results = []

    # --------------------------------------------------------
    # Fold loop
    # --------------------------------------------------------

    for fold_info in folds:

        fold_id = fold_info["fold"]

        train_idx = fold_info["train_idx"]

        test_idx = fold_info["test_idx"]

        X_train_raw = X[train_idx]

        X_test_raw = X[test_idx]

        y_train = y[train_idx]

        y_test = y[test_idx]

        print(
            f"[INFO] Fold {fold_id}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}",
            flush=True
        )

        # ----------------------------------------------------
        # Check classes
        # ----------------------------------------------------

        train_classes = set(
            y_train.tolist()
        )

        if len(train_classes) < 2:

            log_warning(
                f"Fold {fold_id}: "
                f"training hanya memiliki "
                f"{len(train_classes)} kelas. "
                f"Fold dilewati."
            )

            continue

        # ----------------------------------------------------
        # Scaling
        #
        # IMPORTANT:
        # Fit hanya dari training fold.
        # ----------------------------------------------------

        mean, std = fit_scaler(
            X_train_raw
        )

        X_train = transform_features(
            X_train_raw,
            mean,
            std
        )

        X_test = transform_features(
            X_test_raw,
            mean,
            std
        )

        # ----------------------------------------------------
        # Validation split dari TRAIN saja
        #
        # Ini tetap chronological.
        # Tidak menyentuh test fold.
        # ----------------------------------------------------

        validation_size = max(
            1,
            int(
                len(X_train) * 0.15
            )
        )

        if (
            len(X_train)
            - validation_size
            < 10
        ):

            validation_size = 1

        train_end = (
            len(X_train)
            - validation_size
        )

        X_fit = X_train[
            :train_end
        ]

        y_fit = y_train[
            :train_end
        ]

        X_val = X_train[
            train_end:
        ]

        y_val = y_train[
            train_end:
        ]

        # ----------------------------------------------------
        # Check validation classes
        # ----------------------------------------------------

        if len(
            set(y_fit.tolist())
        ) < 2:

            # If validation split causes
            # training to lose a class,
            # use full training data and
            # a copy for validation.

            X_fit = X_train

            y_fit = y_train

            X_val = X_train

            y_val = y_train

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        start_time = time.time()

        classifier = train_classifier(
            X_fit,
            y_fit,
            X_val,
            y_val,
            device,
        )

        training_seconds = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        y_pred = predict(
            classifier,
            X_test,
            device
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics, cm = calculate_metrics(
            y_test,
            y_pred
        )

        print(
            f"        Macro F1           : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"        Balanced Accuracy  : "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"        Accuracy           : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"        MCC                : "
            f"{metrics['mcc']:.4f}"
        )

        print(
            f"        Training time      : "
            f"{training_seconds:.2f}s",
            flush=True
        )

        # ----------------------------------------------------
        # Result row
        # ----------------------------------------------------

        result = {

            "model":
                model_name,

            "aggregation":
                filename,

            "fold":
                fold_id,

            "train_samples":
                len(train_idx),

            "test_samples":
                len(test_idx),

            "n_features":
                X.shape[1],

            "accuracy":
                metrics["accuracy"],

            "balanced_accuracy":
                metrics["balanced_accuracy"],

            "macro_f1":
                metrics["macro_f1"],

            "macro_precision":
                metrics["macro_precision"],

            "macro_recall":
                metrics["macro_recall"],

            "mcc":
                metrics["mcc"],

            "f1_down":
                metrics["f1_down"],

            "f1_flat":
                metrics["f1_flat"],

            "f1_up":
                metrics["f1_up"],

            "precision_down":
                metrics["precision_down"],

            "precision_flat":
                metrics["precision_flat"],

            "precision_up":
                metrics["precision_up"],

            "recall_down":
                metrics["recall_down"],

            "recall_flat":
                metrics["recall_flat"],

            "recall_up":
                metrics["recall_up"],

            "training_seconds":
                training_seconds,
        }

        all_results.append(
            result
        )

        # ----------------------------------------------------
        # Save fold result immediately
        # ----------------------------------------------------

        fold_output = (
            RESULTS_DIR
            /
            f"{model_name}__{filename}"
            f"__fold{fold_id}.json"
        )

        with open(
            fold_output,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2
            )

    return all_results


# ============================================================
# AGGREGATE FOLD RESULTS
# ============================================================

def aggregate_results(
    fold_results
):

    if not fold_results:

        return None

    first = fold_results[0]

    numeric_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "mcc",
        "f1_down",
        "f1_flat",
        "f1_up",
        "precision_down",
        "precision_flat",
        "precision_up",
        "recall_down",
        "recall_flat",
        "recall_up",
        "training_seconds",
    ]

    result = {

        "model":
            first["model"],

        "aggregation":
            first["aggregation"],

        "n_folds":
            len(fold_results),

        "n_features":
            first["n_features"],

    }

    for metric in numeric_metrics:

        values = [
            r[metric]
            for r in fold_results
        ]

        result[
            f"{metric}_mean"
        ] = float(
            np.mean(values)
        )

        result[
            f"{metric}_std"
        ] = float(
            np.std(values)
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    ensure_dirs()

    print()
    print("=" * 70)
    print("TEMPORAL AGGREGATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    calendar = load_calendar()

    log(
        f"Calendar samples: "
        f"{len(calendar):,}"
    )

    log(
        "Direction distribution:"
    )

    print(
        calendar[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    files = discover_aggregation_files()

    log(
        f"Aggregation files found: "
        f"{len(files)}"
    )

    if not files:

        raise ValueError(
            "Tidak ditemukan file A0-A4."
        )

    # --------------------------------------------------------
    # Show plan
    # --------------------------------------------------------

    print()
    print(
        f"Total configurations: "
        f"{len(files)}"
    )

    print(
        f"Validation folds: "
        f"{N_FOLDS}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    print()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_fold_results = []

    all_summary_results = []

    for idx, file_info in enumerate(
        files,
        start=1
    ):

        model_name = file_info["model"]

        filename = file_info["filename"]

        print()
        print(
            f"[{idx}/{len(files)}]"
        )

        print(
            f"Model       : "
            f"{model_name}"
        )

        print(
            f"Aggregation : "
            f"{filename}"
        )

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        summary_file = (
            RESULTS_DIR
            /
            f"{model_name}__{filename}"
            "__summary.json"
        )

        if (
            RESUME
            and summary_file.exists()
        ):

            log(
                "Result sudah ada. "
                "Skip karena RESUME=True."
            )

            with open(
                summary_file,
                "r",
                encoding="utf-8"
            ) as f:

                summary = json.load(f)

            all_summary_results.append(
                summary
            )

            continue

        # ----------------------------------------------------
        # Run evaluation
        # ----------------------------------------------------

        try:

            fold_results = evaluate_one_file(
                file_info,
                calendar,
                device,
            )

            if not fold_results:

                log_warning(
                    "Tidak ada fold yang berhasil."
                )

                continue

            all_fold_results.extend(
                fold_results
            )

            summary = aggregate_results(
                fold_results
            )

            all_summary_results.append(
                summary
            )

            # ------------------------------------------------
            # Save summary immediately
            # ------------------------------------------------

            with open(
                summary_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    summary,
                    f,
                    indent=2
                )

            log(
                f"Finished: "
                f"{model_name} / "
                f"{filename}"
            )

        except Exception as exc:

            log_warning(
                f"FAILED: "
                f"{model_name} / "
                f"{filename}"
            )

            log_warning(
                repr(exc)
            )

    # ========================================================
    # SAVE ALL FOLD RESULTS
    # ========================================================

    if all_fold_results:

        fold_df = pd.DataFrame(
            all_fold_results
        )

        fold_df.to_csv(
            OUTPUT_DIR
            /
            "temporal_aggregation_fold_results.csv",
            index=False,
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    if all_summary_results:

        summary_df = pd.DataFrame(
            all_summary_results
        )

        summary_df = summary_df.sort_values(
            "macro_f1_mean",
            ascending=False
        ).reset_index(
            drop=True
        )

        summary_df.insert(
            0,
            "rank",
            np.arange(
                1,
                len(summary_df) + 1
            )
        )

        summary_df.to_csv(
            SUMMARY_CSV,
            index=False
        )

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        ranking_columns = [
            "rank",
            "model",
            "aggregation",
            "n_folds",
            "n_features",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "accuracy_mean",
            "accuracy_std",
            "mcc_mean",
            "mcc_std",
            "f1_down_mean",
            "f1_flat_mean",
            "f1_up_mean",
        ]

        ranking_columns = [
            col
            for col in ranking_columns
            if col in summary_df.columns
        ]

        ranking_df = summary_df[
            ranking_columns
        ].copy()

        ranking_df.to_csv(
            RANKING_CSV,
            index=False
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION FINISHED")
    print("=" * 70)

    if all_summary_results:

        print()
        print(
            "TOP 10 CONFIGURATIONS "
            "BERDASARKAN MEAN MACRO F1"
        )

        top10 = (
            pd.DataFrame(
                all_summary_results
            )
            .sort_values(
                "macro_f1_mean",
                ascending=False
            )
            .head(10)
        )

        display_columns = [
            "model",
            "aggregation",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "accuracy_mean",
            "mcc_mean",
        ]

        print(
            top10[
                display_columns
            ].to_string(
                index=False
            )
        )

        print()
        print(
            f"Full results:\n"
            f"{SUMMARY_CSV}"
        )

        print(
            f"Ranking:\n"
            f"{RANKING_CSV}"
        )

    else:

        log_warning(
            "Tidak ada hasil evaluasi."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()