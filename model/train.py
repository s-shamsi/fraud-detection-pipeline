"""
XGBoost Fraud Detection Model – Training
==========================================
Trains a gradient-boosted tree classifier on the synthetic transaction dataset,
with stratified k-fold cross-validation, class-weight balancing, and exports
artefacts ready for ONNX conversion.

GCP note: For production, swap the local Parquet read for a BigQuery query via
Vertex AI Workbench or a Vertex AI Training Job. The rest of the code is
cloud-agnostic.

Usage:
    python model/train.py
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import from sibling package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate_data import get_feature_columns

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/transactions.parquet"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# ── Hyper-parameters ────────────────────────────────────────────────────────────

XGBOOST_PARAMS = {
    "n_estimators":      600,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "binary:logistic",
    "eval_metric":       "aucpr",
    "tree_method":       "hist",         # GPU: change to "gpu_hist"
    "random_state":      42,
    "n_jobs":            -1,
}

N_FOLDS       = 5
FRAUD_COL     = "is_fraud"
THRESHOLD     = 0.40   # Tuned for high recall on fraud
SCALE_POS_WT  = 60     # ≈ 1/fraud_rate; compensates for class imbalance


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    print(f"Loading data from {DATA_PATH} …")
    df = pd.read_parquet(DATA_PATH)

    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[FRAUD_COL].values.astype(np.int32)

    print(f"  Shape  : {X.shape}")
    print(f"  Fraud  : {y.mean():.2%}  ({y.sum():,} / {len(y):,})")
    return df, X, y


def cross_validate(X: np.ndarray, y: np.ndarray) -> dict:
    """Stratified k-fold CV to assess generalisation before final fit."""
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = {"auc_roc": [], "auc_pr": [], "f1": []}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        clf = xgb.XGBClassifier(
            **XGBOOST_PARAMS,
            scale_pos_weight=SCALE_POS_WT,
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= THRESHOLD).astype(int)

        scores["auc_roc"].append(roc_auc_score(y_val, probs))
        scores["auc_pr"].append(average_precision_score(y_val, probs))
        scores["f1"].append(f1_score(y_val, preds))

        print(
            f"  Fold {fold + 1}/{N_FOLDS}"
            f"  AUC-ROC={scores['auc_roc'][-1]:.4f}"
            f"  AUC-PR={scores['auc_pr'][-1]:.4f}"
            f"  F1={scores['f1'][-1]:.4f}"
        )

    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores.items()}
    print("\n── Cross-validation summary ─────────────────────────────────────")
    for metric, s in summary.items():
        print(f"  {metric:<12}  {s['mean']:.4f} ± {s['std']:.4f}")
    return summary


def train_final_model(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """Train final model on the complete dataset."""
    print("\n── Training final model on full dataset …")
    clf = xgb.XGBClassifier(
        **XGBOOST_PARAMS,
        scale_pos_weight=SCALE_POS_WT,
        early_stopping_rounds=None,
    )
    clf.fit(X, y, verbose=100)
    return clf


def evaluate_final_model(
    clf: xgb.XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    metrics = {
        "auc_roc":   float(roc_auc_score(y, probs)),
        "auc_pr":    float(average_precision_score(y, probs)),
        "f1_binary": float(f1_score(y, preds)),
        "threshold": THRESHOLD,
    }
    print("\n── Final model evaluation (train set, indicative only) ──────────")
    for k, v in metrics.items():
        print(f"  {k:<16} {v}")
    print("\n" + classification_report(y, preds, target_names=["legit", "fraud"]))
    return metrics


def save_feature_importance(clf: xgb.XGBClassifier, feature_names: list[str]) -> None:
    importance = clf.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
    )
    fi_df.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    print("\n── Top 10 features ──────────────────────────────────────────────")
    print(fi_df.head(10).to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    df, X, y = load_data()
    feature_cols = get_feature_columns()

    # 1. Cross-validate
    print("\n── Cross-validation ─────────────────────────────────────────────")
    cv_metrics = cross_validate(X, y)

    # 2. Train final model
    clf = train_final_model(X, y)

    # 3. Evaluate
    final_metrics = evaluate_final_model(clf, X, y)

    # 4. Save XGBoost native model (used as ONNX conversion source)
    clf.save_model(str(ARTIFACT_DIR / "fraud_detector.xgb.json"))
    joblib.dump(clf, ARTIFACT_DIR / "fraud_detector.pkl")
    print(f"\nModel saved to {ARTIFACT_DIR}/fraud_detector.xgb.json")

    # 5. Save metadata
    metadata = {
        "feature_columns":  feature_cols,
        "n_features":       len(feature_cols),
        "threshold":        THRESHOLD,
        "xgboost_params":   XGBOOST_PARAMS,
        "cv_metrics":       cv_metrics,
        "final_metrics":    final_metrics,
        "model_version":    "1.0.0",
    }
    with open(ARTIFACT_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 6. Feature importance
    save_feature_importance(clf, feature_cols)

    print(f"\n✅ Training complete. Artefacts written to {ARTIFACT_DIR}/")
    print("   Next step: python model/convert_to_onnx.py")


if __name__ == "__main__":
    main()
