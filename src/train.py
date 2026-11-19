"""Illustrative predictive modeling for 30-day diabetes readmission.

The original collaborative graduate course project used JMP. This Python
implementation is a separate companion; it does not reproduce those metrics.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
CATEGORICAL = [
    "age", "race", "gender", "admission_type_id",
    "discharge_disposition_id", "diabetesMed", "change",
    "A1Cresult", "max_glu_serum",
]
FEATURES = NUMERIC + CATEGORICAL


def age_lower_bound(value: object) -> int:
    """Parse the lower bound of the source dataset's age bands."""
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else -1


def load_uci_frame(path: Path) -> pd.DataFrame:
    """Read the public UCI diabetes dataset and define the 30-day outcome."""
    raw = pd.read_csv(path, na_values=["?"], low_memory=False)
    required = set(FEATURES + ["readmitted"])
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Input is missing required UCI columns: {missing}")

    # Keep one encounter per patient when patient identifiers are present.
    if "patient_nbr" in raw.columns:
        raw = raw.drop_duplicates(subset=["patient_nbr"], keep="first")
    raw = raw.loc[raw["age"].map(age_lower_bound) >= 20].copy()
    raw["readmit_30d"] = (raw["readmitted"] == "<30").astype(int)
    return raw[FEATURES + ["readmit_30d"]]


def make_demo_frame() -> pd.DataFrame:
    """Generate fully synthetic features and labels for a runnable example."""
    X, y = make_classification(
        n_samples=1300, n_features=len(NUMERIC), n_informative=6,
        n_redundant=1, weights=[0.91, 0.09], flip_y=0.04,
        random_state=31,
    )
    frame = pd.DataFrame(X, columns=NUMERIC)
    for field in CATEGORICAL:
        frame[field] = np.where(
            np.arange(len(frame)) % 3 == 0, "Category A", "Category B"
        )
    frame["readmit_30d"] = y
    return frame


def make_preprocessor() -> ColumnTransformer:
    """Fit all transforms within the training pipeline to avoid split leakage."""
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ],
        remainder="drop",
    )


def lift_at_fraction(
    y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.10
) -> float:
    """Top-decile ranking lift (NOT the original project's lift formula)."""
    k = max(1, int(np.ceil(len(y_true) * fraction)))
    baseline = float(np.mean(y_true))
    if baseline == 0:
        return float("nan")
    top_indices = np.argsort(scores)[-k:]
    return float(np.mean(y_true[top_indices]) / baseline)


def run(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare three models on a held-out, stratified 30-day readmission task."""
    frame = frame.copy()
    for field in NUMERIC:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    for field in CATEGORICAL:
        frame[field] = frame[field].astype("string").fillna("Unknown").astype(str)
    y = frame["readmit_30d"].astype(int).to_numpy()
    X = frame[FEATURES]
    if len(np.unique(y)) != 2:
        raise ValueError("Binary evaluation requires both classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    models = {
        "Logistic regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced_subsample",
            n_jobs=2, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=90, max_depth=4, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=2, random_state=42,
        ),
    }
    results = []
    for name, model in models.items():
        pipeline = Pipeline([("prep", make_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        scores = pipeline.predict_proba(X_test)[:, 1]
        predicted = (scores >= 0.5).astype(int)
        results.append({
            "model": name,
            "roc_auc": round(float(roc_auc_score(y_test, scores)), 3),
            "avg_precision": round(float(average_precision_score(y_test, scores)), 3),
            "precision_0.5": round(
                float(precision_score(y_test, predicted, zero_division=0)), 3
            ),
            "recall_0.5": round(
                float(recall_score(y_test, predicted, zero_division=0)), 3
            ),
            "lift_top_10pct": round(lift_at_fraction(y_test, scores), 3),
        })
    print("Test-set positive rate:", round(float(y_test.mean()), 3))
    results_frame = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print(results_frame.to_string(index=False))
    return results_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use synthetic sample data")
    source.add_argument("--csv", type=Path, help="Path to public UCI diabetic_data.csv")
    args = parser.parse_args()
    if args.demo:
        print("DEMO MODE: synthetic toy data; performance is illustrative only.")
    frame = make_demo_frame() if args.demo else load_uci_frame(args.csv)
    run(frame)


if __name__ == "__main__":
    main()
