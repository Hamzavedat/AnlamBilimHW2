import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RUN_NAME = "hw2_full_seed42"
FEATURES_PATH = os.path.join("runs", RUN_NAME, "data", "features.csv")
RESULTS_DIR = os.path.join("runs", RUN_NAME, "results")
SEED = 42
TEST_SIZE = 0.2


def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(value) for value in obj]
    return obj


def evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["random_forest"] = {
        "accuracy": float(accuracy_score(y_test, rf_pred)),
        "macro_f1": float(f1_score(y_test, rf_pred, average="macro")),
        "classification_report": classification_report(y_test, rf_pred, output_dict=True),
    }

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["logistic_regression"] = {
        "accuracy": float(accuracy_score(y_test, lr_pred)),
        "macro_f1": float(f1_score(y_test, lr_pred, average="macro")),
        "classification_report": classification_report(y_test, lr_pred, output_dict=True),
    }

    return convert_numpy_types(results)


def grouped_split_with_class_coverage(df, group_col):
    all_labels = set(df["category"].unique())
    splitter = GroupShuffleSplit(n_splits=50, test_size=TEST_SIZE, random_state=SEED)

    for train_idx, test_idx in splitter.split(df, y=df["category"], groups=df[group_col]):
        train_labels = set(df.iloc[train_idx]["category"].unique())
        test_labels = set(df.iloc[test_idx]["category"].unique())
        if train_labels == all_labels and test_labels == all_labels:
            return train_idx, test_idx

    raise ValueError("Could not find a grouped split that contains all categories in train and test.")


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("features.csv not found. Run extract_features.py first.")

    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} trajectories.")

    raw_features = [col for col in df.columns if col.startswith("raw_geo_")]
    normalized_features = [col for col in df.columns if col.startswith("norm_geo_") or col.startswith("si_")]

    X_raw = df[raw_features]
    X_norm = df[normalized_features]

    y_model = df["model_name"]
    X_train_raw, X_test_raw, y_train_model, y_test_model = train_test_split(
        X_raw,
        y_model,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_model,
    )
    X_train_norm, X_test_norm, _, _ = train_test_split(
        X_norm,
        y_model,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_model,
    )

    model_raw_results = evaluate_models(X_train_raw, X_test_raw, y_train_model, y_test_model)
    model_norm_results = evaluate_models(X_train_norm, X_test_norm, y_train_model, y_test_model)

    group_col = "prompt_id"
    train_idx, test_idx = grouped_split_with_class_coverage(df, group_col)

    category_raw_results = evaluate_models(
        X_raw.iloc[train_idx],
        X_raw.iloc[test_idx],
        df["category"].iloc[train_idx],
        df["category"].iloc[test_idx],
    )
    category_norm_results = evaluate_models(
        X_norm.iloc[train_idx],
        X_norm.iloc[test_idx],
        df["category"].iloc[train_idx],
        df["category"].iloc[test_idx],
    )

    metrics = {
        "run": {
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "test_size": TEST_SIZE,
            "features_path": os.path.abspath(FEATURES_PATH),
            "num_rows": int(len(df)),
        },
        "task_model_identity": {
            "split": "stratified train_test_split",
            "raw_features": model_raw_results,
            "normalized_features": model_norm_results,
        },
        "task_category_prediction_grouped": {
            "split": "GroupShuffleSplit by prompt_id",
            "coverage_satisfied": True,
            "raw_features": category_raw_results,
            "normalized_features": category_norm_results,
        },
    }

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_lines = [
        "# Experiment Report",
        "",
        f"- Rows: {len(df)}",
        "",
        "## Model Identity",
        f"- Raw RF accuracy: {model_raw_results['random_forest']['accuracy']:.4f}",
        f"- Raw LR accuracy: {model_raw_results['logistic_regression']['accuracy']:.4f}",
        f"- Normalized RF accuracy: {model_norm_results['random_forest']['accuracy']:.4f}",
        f"- Normalized LR accuracy: {model_norm_results['logistic_regression']['accuracy']:.4f}",
        "",
        "## Category Prediction (Grouped)",
        f"- Raw RF accuracy: {category_raw_results['random_forest']['accuracy']:.4f}",
        f"- Raw LR accuracy: {category_raw_results['logistic_regression']['accuracy']:.4f}",
        f"- Normalized RF accuracy: {category_norm_results['random_forest']['accuracy']:.4f}",
        f"- Normalized LR accuracy: {category_norm_results['logistic_regression']['accuracy']:.4f}",
        "- Coverage satisfied: True",
    ]

    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Saved {metrics_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
