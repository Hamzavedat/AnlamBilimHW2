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

    feature_sets = {
        "human_raw_geometry": [col for col in df.columns if col.startswith("raw_geo_")],
        "human_normalized_geometry": [
            col for col in df.columns if col.startswith("norm_geo_") or col.startswith("si_")
        ],
        "ai_pca_projection": [col for col in df.columns if col.startswith("within_model_pca_")],
    }
    feature_sets["ai_combined_geometry"] = (
        feature_sets["human_normalized_geometry"] + feature_sets["ai_pca_projection"]
    )

    y_model = df["model_name"]
    group_col = "prompt_id"
    train_idx, test_idx = grouped_split_with_class_coverage(df, group_col)

    model_identity_results = {}
    category_results = {}

    for feature_set_name, feature_cols in feature_sets.items():
        X = df[feature_cols]

        X_train, X_test, y_train_model, y_test_model = train_test_split(
            X,
            y_model,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y_model,
        )
        model_identity_results[feature_set_name] = evaluate_models(
            X_train,
            X_test,
            y_train_model,
            y_test_model,
        )

        category_results[feature_set_name] = evaluate_models(
            X.iloc[train_idx],
            X.iloc[test_idx],
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
        "feature_sets": {
            name: {
                "source": "human" if name.startswith("human_") else "ai_assisted",
                "num_features": len(cols),
                "columns": cols,
            }
            for name, cols in feature_sets.items()
        },
        "task_model_identity": {
            "split": "stratified train_test_split",
            "results": model_identity_results,
        },
        "task_category_prediction_grouped": {
            "split": "GroupShuffleSplit by prompt_id",
            "coverage_satisfied": True,
            "results": category_results,
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
        "## Four Trajectory Inferences",
        "- Human 1: raw geometric features",
        "- Human 2: normalized and scale-invariant geometric features",
        "- AI-assisted 1: within-model PCA projection features",
        "- AI-assisted 2: combined normalized geometry + PCA projection features",
        "",
        "## Model Identity",
    ]

    for feature_set_name, results in model_identity_results.items():
        report_lines.extend(
            [
                f"- {feature_set_name} RF accuracy: {results['random_forest']['accuracy']:.4f}",
                f"- {feature_set_name} LR accuracy: {results['logistic_regression']['accuracy']:.4f}",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Category Prediction (Grouped)",
        ]
    )

    for feature_set_name, results in category_results.items():
        report_lines.extend(
            [
                f"- {feature_set_name} RF accuracy: {results['random_forest']['accuracy']:.4f}",
                f"- {feature_set_name} LR accuracy: {results['logistic_regression']['accuracy']:.4f}",
            ]
        )

    report_lines.extend(
        [
            "- Coverage satisfied: True",
        ]
    )

    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Saved {metrics_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
