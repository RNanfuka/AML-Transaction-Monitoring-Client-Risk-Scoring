from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import AMLFeatureEngineer


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    pr_auc: float
    roc_auc: float


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                AMLFeatureEngineer.numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
                AMLFeatureEngineer.categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "hist_gradient_boosting": Pipeline([
            ("prep", _preprocessor()),
            ("model", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                class_weight="balanced", early_stopping=False, random_state=random_state)),
        ]),
        "logistic_regression": Pipeline(
            steps=[
                ("prep", _preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", _preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=180,
                        max_depth=18,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_probability_model(
    y_true: pd.Series,
    prob: np.ndarray,
) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return None, None
    return float(average_precision_score(y_true, prob)), float(roc_auc_score(y_true, prob))


def choose_threshold(y_true, prob, target_recall=0.80):
    """Exact score boundaries; minimize investigator alerts at target recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1].")
    y = np.asarray(y_true)
    prob = np.asarray(prob)
    if len(y) == 0 or not np.isin(y, [0, 1]).all() or y.sum() == 0:
        raise ValueError("Threshold selection requires labelled positives.")
    order = np.argsort(-prob, kind="stable")
    scores, labels = prob[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
    tp = np.cumsum(labels)[ends]
    alerts = ends + 1
    fp, fn = alerts-tp, y.sum()-tp
    tn = len(y)-y.sum()-fp
    curve = pd.DataFrame({
        "threshold": scores[ends], "precision": tp/alerts, "recall": tp/y.sum(),
        "f1": 2*tp/(2*tp+fp+fn), "false_positive_rate": fp/max(len(y)-y.sum(), 1),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "true_negatives": tn, "number_of_alerts": alerts, "alert_rate": alerts/len(y),
    })
    zero = summarize_threshold(y, prob, float(np.nextafter(scores[0], np.inf)))
    curve = pd.concat([pd.DataFrame([zero]), curve], ignore_index=True)
    best = curve[curve.recall >= target_recall].sort_values(
        ["number_of_alerts", "false_positive_rate", "threshold"], ascending=[True, True, False]
    ).iloc[0]
    return float(best.threshold), curve.sort_values("threshold").reset_index(drop=True)


def summarize_threshold(
    y_true: pd.Series,
    prob: np.ndarray,
    threshold: float,
) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "number_of_alerts": int(pred.sum()),
        "alert_rate": float(pred.mean()),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def save_model_bundle(
    path: str | Path,
    pipeline: Pipeline,
    feature_engineer: AMLFeatureEngineer,
    threshold: float,
    model_name: str,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_engineer": feature_engineer,
            "threshold": threshold,
            "model_name": model_name,
        },
        path,
    )


def save_metrics(path: str | Path, metrics: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def feature_importance(models):
    """Signed standardized coefficients and impurity importance, where supported."""
    rows = []
    for name, pipeline in models.items():
        estimator = pipeline.named_steps["model"]
        names = pipeline.named_steps["prep"].get_feature_names_out()
        if hasattr(estimator, "coef_"):
            values, method = estimator.coef_[0], "standardized coefficient"
        elif hasattr(estimator, "feature_importances_"):
            values, method = estimator.feature_importances_, "impurity importance"
        else:
            continue
        rows.extend({"model": name, "feature": f, "importance": float(v), "method": method}
                    for f, v in zip(names, values))
    return pd.DataFrame(rows)
