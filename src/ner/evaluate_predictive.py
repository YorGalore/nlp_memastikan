"""Evaluate daily NER features as additions to lagged JISDOR predictors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
ALIGNED_FILE = DATA_DIR / "aligned_daily.csv"
NER_FILE = DATA_DIR / "ner" / "daily_entity_features.csv"
OUTPUT_DIR = DATA_DIR / "ner" / "predictive_evaluation"
CLASS_ORDER = ("low", "medium", "high")
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
RANDOM_STATE = 42

# Every feature is known before the prediction day.  Current JISDOR price,
# return, and realized volatility are intentionally excluded to prevent leakage.
JISDOR_FEATURES = [
    "jisdor_lag1_log_return", "jisdor_lag2_log_return",
    "jisdor_lag3_log_return", "jisdor_lag4_log_return",
    "jisdor_known4_mean", "jisdor_known4_std",
    "jisdor_vol_if_r0", "jisdor_vol_if_rtyp_pos",
    "jisdor_vol_if_rtyp_neg", "jisdor_lag1_volatility_class",
]


def load_targets() -> pd.DataFrame:
    if not ALIGNED_FILE.exists():
        raise FileNotFoundError(f"JISDOR-aligned data tidak ditemukan: {ALIGNED_FILE}")
    target = pd.read_csv(ALIGNED_FILE)
    required = {"date", "log_return"}
    missing = required - set(target.columns)
    if missing:
        raise ValueError(f"aligned_daily.csv tidak memiliki kolom: {sorted(missing)}")
    target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.normalize()
    target["log_return"] = pd.to_numeric(target["log_return"], errors="coerce")
    target = target.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if "realized_vol" not in target.columns:
        target["realized_vol"] = target["log_return"].rolling(5, min_periods=5).std()
    target["realized_vol"] = pd.to_numeric(target["realized_vol"], errors="coerce")

    labels = pd.Series(np.nan, index=target.index, dtype="object")
    if "volatility_class" in target.columns:
        labels = target["volatility_class"].astype("string").str.lower().str.strip()
        labels = labels.where(labels.isin(CLASS_ORDER), np.nan)
    if labels.isna().any():
        calibration = target.loc[target["realized_vol"].notna(), "realized_vol"]
        calibration = calibration.iloc[:max(3, int(len(calibration) * 0.70))]
        low, high = calibration.quantile([1 / 3, 2 / 3]).to_numpy()
        labels = pd.cut(target["realized_vol"], [-np.inf, low, high, np.inf], labels=CLASS_ORDER).astype("object")
    target["volatility_class"] = labels
    target = target.dropna(subset=["realized_vol", "volatility_class"]).reset_index(drop=True)

    returns = target["log_return"]
    known = pd.concat([returns.shift(lag) for lag in (1, 2, 3, 4)], axis=1)
    for lag in (1, 2, 3, 4):
        target[f"jisdor_lag{lag}_log_return"] = returns.shift(lag)
    target["jisdor_known4_mean"] = known.mean(axis=1, skipna=False)
    target["jisdor_known4_std"] = known.std(axis=1, skipna=False)

    def volatility_if(return_at_t: np.ndarray) -> np.ndarray:
        return np.std(np.column_stack([known.to_numpy(), return_at_t]), axis=1, ddof=1)

    typical = returns.abs().expanding().median().shift(1).to_numpy()
    target["jisdor_vol_if_r0"] = volatility_if(np.zeros(len(target)))
    target["jisdor_vol_if_rtyp_pos"] = volatility_if(typical)
    target["jisdor_vol_if_rtyp_neg"] = volatility_if(-typical)
    target["jisdor_lag1_volatility_class"] = target["volatility_class"].map(
        {label: index for index, label in enumerate(CLASS_ORDER)}
    ).shift(1)
    return target


def load_ner_features() -> pd.DataFrame:
    if not NER_FILE.exists():
        raise FileNotFoundError(f"Fitur NER tidak ditemukan: {NER_FILE}")
    ner = pd.read_csv(NER_FILE)
    if "date" not in ner.columns:
        raise ValueError("daily_entity_features.csv tidak memiliki kolom date")
    ner["date"] = pd.to_datetime(ner["date"], errors="coerce").dt.normalize()
    ner = ner.dropna(subset=["date"])
    feature_cols = [c for c in ner.columns if c != "date" and pd.api.types.is_numeric_dtype(ner[c])]
    if not feature_cols:
        raise ValueError("Tidak ada fitur NER numerik pada daily_entity_features.csv")
    ner = ner[["date", *feature_cols]].copy()
    ner[feature_cols] = ner[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return ner.groupby("date", as_index=False)[feature_cols].mean()


def classifiers() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", XGBClassifier(
                objective="multi:softprob", num_class=3, n_estimators=300,
                max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }


def evaluate(X: np.ndarray, y: np.ndarray, dates: pd.Series, scenario: str) -> list[dict]:
    rows: list[dict] = []
    class_to_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    train_end = int(len(X) * TRAIN_FRAC)
    valid_end = train_end + int(len(X) * VALID_FRAC)
    train_idx = np.arange(train_end)
    y_train = y[train_idx]
    if len(set(y_train)) < 2:
        raise ValueError("Split train tidak memiliki minimal dua kelas.")
    encoded_train = np.asarray([class_to_index[label] for label in y_train], dtype=int)
    for classifier_name, classifier in classifiers().items():
        classifier.fit(X[train_idx], encoded_train)
        for evaluation_split, eval_idx in {
            "validation": np.arange(train_end, valid_end), "test": np.arange(valid_end, len(X)),
        }.items():
            y_eval = y[eval_idx]
            prediction = classifier.predict(X[eval_idx]).astype(int)
            prediction = np.asarray([CLASS_ORDER[index] for index in prediction])
            rows.append({
                "scenario": scenario, "classifier": classifier_name, "evaluation_split": evaluation_split,
                "train_start": str(dates.iloc[train_idx].min().date()), "train_end": str(dates.iloc[train_idx].max().date()),
                "evaluation_start": str(dates.iloc[eval_idx].min().date()), "evaluation_end": str(dates.iloc[eval_idx].max().date()),
                "n_train": len(train_idx), "n_validation": valid_end - train_end, "n_test": len(X) - valid_end, "n_features": X.shape[1],
                "accuracy": accuracy_score(y_eval, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_eval, prediction),
                "macro_f1": f1_score(y_eval, prediction, labels=CLASS_ORDER, average="macro", zero_division=0),
            })
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    ner = load_ner_features()
    y = targets["volatility_class"].astype(str).to_numpy()
    dates = targets["date"].reset_index(drop=True)
    price = targets[JISDOR_FEATURES].to_numpy(dtype=np.float32)
    merged = targets[["date"]].merge(ner, on="date", how="left")
    ner_cols = [c for c in ner.columns if c != "date"]

    rows = evaluate(price, y, dates, "jisdor_price_only")
    rows.extend(evaluate(np.hstack([price, merged[ner_cols].fillna(0.0).to_numpy(dtype=np.float32)]), y, dates, "jisdor_plus_ner"))
    fold_df = pd.DataFrame(rows)
    summary = fold_df.groupby(["scenario", "classifier", "evaluation_split"], as_index=False).agg(
        n_evaluations=("scenario", "count"), n_features=("n_features", "first"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), macro_f1_mean=("macro_f1", "mean"),
    ).sort_values("macro_f1_mean", ascending=False)
    summary["accuracy_std"] = summary["accuracy_std"].fillna(0.0)
    fold_df.to_csv(OUTPUT_DIR / "ner_jisdor_fold_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "ner_jisdor_summary.csv", index=False)
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump({"jisdor_features": JISDOR_FEATURES, "n_ner_features": len(ner_cols),
                   "classifiers": ["logistic_regression", "xgboost"],
                   "scenarios": ["jisdor_price_only", "jisdor_plus_ner"],
                   "split": "chronological fixed split: 70% train, 15% validation, 15% test"}, file, indent=2)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
