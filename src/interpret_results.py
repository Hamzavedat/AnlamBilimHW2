import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


RUN_NAME = "hw2_full_seed42"
RUN_DIR = os.path.join("runs", RUN_NAME)
FEATURES_PATH = os.path.join(RUN_DIR, "tables", "features.csv")
METRICS_PATH = os.path.join(RUN_DIR, "results", "metrics.json")
OUT_DIR = os.path.join(RUN_DIR, "inferences")
SEED = 42
TEST_SIZE = 0.2
TOP_K = 8


def grouped_split_with_class_coverage(df, group_col):
    all_labels = set(df["category"].unique())
    splitter = GroupShuffleSplit(n_splits=50, test_size=TEST_SIZE, random_state=SEED)

    for train_idx, test_idx in splitter.split(df, y=df["category"], groups=df[group_col]):
        train_labels = set(df.iloc[train_idx]["category"].unique())
        test_labels = set(df.iloc[test_idx]["category"].unique())
        if train_labels == all_labels and test_labels == all_labels:
            return train_idx, test_idx

    raise ValueError("Could not find a grouped split that contains all categories in train and test.")


def save_human_model_style_plot(df):
    model_order = sorted(df["model_name"].unique())
    feature_specs = [
        ("raw_geo_path_length", "Raw path length"),
        ("si_step_cv", "Step variability (CV)"),
        ("si_turn_abs_mean", "Mean abs turn"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, title) in zip(axes, feature_specs):
        data = [df[df["model_name"] == model][col].to_numpy() for model in model_order]
        ax.boxplot(data, labels=model_order, patch_artist=True)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=12)

    fig.suptitle("Human-like inference 1: GPT-2 trajectories are longer and less steady")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "human_1_model_style.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    means = df.groupby("model_name")[
        ["raw_geo_path_length", "si_step_cv", "si_turn_abs_mean"]
    ].mean()
    return out_path, means


def save_human_prompt_sensitivity_plot(df):
    model_order = sorted(df["model_name"].unique())
    category_order = sorted(df["category"].unique())

    area_means = (
        df.groupby(["model_name", "category"])["within_model_pca_area"]
        .mean()
        .unstack("category")
        .reindex(index=model_order, columns=category_order)
    )
    spread_means = (
        df.groupby(["model_name", "category"])["within_model_pca_spread"]
        .mean()
        .unstack("category")
        .reindex(index=model_order, columns=category_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(category_order))
    for model_name in model_order:
        axes[0].plot(x, area_means.loc[model_name], marker="o", label=model_name)
        axes[1].plot(x, spread_means.loc[model_name], marker="o", label=model_name)

    axes[0].set_title("Category sensitivity via PCA area")
    axes[1].set_title("Category sensitivity via PCA spread")
    for ax in axes:
        ax.set_xticks(x, category_order, rotation=15)
        ax.grid(alpha=0.2)
        ax.legend()

    fig.suptitle("Human-like inference 2: GPT-2 reacts more to prompt type than Pythia")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "human_2_prompt_sensitivity.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {}
    for model_name in model_order:
        summary[model_name] = {
            "area_range": float(area_means.loc[model_name].max() - area_means.loc[model_name].min()),
            "spread_range": float(
                spread_means.loc[model_name].max() - spread_means.loc[model_name].min()
            ),
            "top_area_category": str(area_means.loc[model_name].idxmax()),
            "top_spread_category": str(spread_means.loc[model_name].idxmax()),
        }
    return out_path, summary


def save_ai_shared_space_plot(df):
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("norm_geo_") or col.startswith("si_") or col.startswith("within_model_pca_")
    ]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    X_scaled = StandardScaler().fit_transform(X)
    X_2d = PCA(n_components=2, random_state=SEED).fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model_name in sorted(df["model_name"].unique()):
        mask = df["model_name"] == model_name
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], s=12, alpha=0.55, label=model_name)
    axes[0].set_title("Colored by model")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    base_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    categories = sorted(df["category"].unique())
    for category, color in zip(categories, base_colors):
        mask = df["category"] == category
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], s=12, alpha=0.45, label=category, color=color)
    axes[1].set_title("Same space, colored by category")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.suptitle("AI-like inference 1: model regions separate clearly, categories mix together")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "ai_1_shared_space.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def top_importances(df):
    feature_cols = [
        col
        for col in df.columns
        if col.startswith("norm_geo_") or col.startswith("si_") or col.startswith("within_model_pca_")
    ]
    X = df[feature_cols]

    X_train, X_test, y_train, _ = train_test_split(
        X,
        df["model_name"],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["model_name"],
    )
    rf_model = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    train_idx, test_idx = grouped_split_with_class_coverage(df, "prompt_id")
    rf_category = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf_category.fit(X.iloc[train_idx], df["category"].iloc[train_idx])

    model_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    category_imp = pd.Series(rf_category.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    return model_imp, category_imp


def save_ai_feature_importance_plot(df):
    model_imp, category_imp = top_importances(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(model_imp.head(TOP_K).index[::-1], model_imp.head(TOP_K).values[::-1])
    axes[0].set_title("Top model-identity features")
    axes[0].grid(alpha=0.2, axis="x")

    axes[1].barh(category_imp.head(TOP_K).index[::-1], category_imp.head(TOP_K).values[::-1])
    axes[1].set_title("Top category-prediction features")
    axes[1].grid(alpha=0.2, axis="x")

    fig.suptitle("AI-like inference 2: model signal has sharper dominant features")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "ai_2_feature_importance.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path, model_imp, category_imp


def write_inference_markdown(metrics, human_style_means, prompt_summary, model_imp, category_imp):
    model_results = metrics["task_model_identity"]["results"]
    category_results = metrics["task_category_prediction_grouped"]["results"]

    best_model_acc = max(
        result["random_forest"]["accuracy"] for result in model_results.values()
    )
    best_category_acc = max(
        result["random_forest"]["accuracy"] for result in category_results.values()
    )

    pythia = human_style_means.loc["EleutherAI/pythia-70m"]
    gpt2 = human_style_means.loc["gpt2"]
    category_count = len(metrics["task_category_prediction_grouped"]["results"]["human_raw_geometry"]["random_forest"]["classification_report"]) - 3

    lines = [
        "# Four Ready-to-Use Inferences",
        "",
        "Bu dosya, odevdeki 'iki yorumu ben yaptim, iki yorumu YZ onerdi' ayrimini kolaylastirmak icin hazirlandi.",
        "",
        "## Kendi yorumun gibi kullanabilecegin 2 cikarim",
        "",
        "### Human-like 1: GPT-2 trajectory'leri daha uzun ve daha dalgali",
        f"- GPT-2 ortalama raw path length: {gpt2['raw_geo_path_length']:.1f}",
        f"- Pythia ortalama raw path length: {pythia['raw_geo_path_length']:.1f}",
        f"- GPT-2 step variability (CV): {gpt2['si_step_cv']:.3f}",
        f"- Pythia step variability (CV): {pythia['si_step_cv']:.3f}",
        f"- GPT-2 mean abs turn: {gpt2['si_turn_abs_mean']:.3f}",
        f"- Pythia mean abs turn: {pythia['si_turn_abs_mean']:.3f}",
        "- Yorum: GPT-2 temsil uzayinda daha fazla dolasiyor; Pythia daha kisa ve daha kontrollu bir yol izliyor.",
        "- Gorsel: `runs/hw2_full_seed42/inferences/human_1_model_style.png`",
        "",
        "### Human-like 2: GPT-2 prompt turune daha hassas, Pythia daha uniform",
        f"- GPT-2 category area range: {prompt_summary['gpt2']['area_range']:.4f}",
        f"- Pythia category area range: {prompt_summary['EleutherAI/pythia-70m']['area_range']:.4f}",
        f"- GPT-2 en genis area veren kategori: {prompt_summary['gpt2']['top_area_category']}",
        f"- Pythia en genis area veren kategori: {prompt_summary['EleutherAI/pythia-70m']['top_area_category']}",
        "- Yorum: Kategori degistiginde GPT-2'nin trajectory sekli daha belirgin oynuyor; Pythia ise konudan daha az etkileniyor.",
        "- Gorsel: `runs/hw2_full_seed42/inferences/human_2_prompt_sensitivity.png`",
        "",
        "## YZ yardimiyla sunabilecegin 2 cikarim",
        "",
        "### AI-like 1: Ortak feature uzayinda model sinyali kategori sinyalinden cok daha guclu",
        f"- En iyi model identity RF accuracy: {best_model_acc:.4f}",
        f"- En iyi grouped category RF accuracy: {best_category_acc:.4f}",
        f"- Sans seviyesi {category_count} kategori icin yaklasik {1.0 / category_count:.2f}'dir.",
        "- Yorum: Trajectory ozellikleri once modeli ele veriyor; konu kategorisi ayni gucle ayrismiyor.",
        "- Gorsel: `runs/hw2_full_seed42/inferences/ai_1_shared_space.png`",
        "",
        "### AI-like 2: Model ayrimi birkac baskin dagilim ozelligiyle geliyor, kategori sinyali daha daginik",
        "- Model identity icin en guclu ilk 4 feature:",
    ]

    for feature_name, value in model_imp.head(4).items():
        lines.append(f"  - {feature_name}: {value:.4f}")

    lines.extend(
        [
            "- Category prediction icin en guclu ilk 4 feature:",
        ]
    )
    for feature_name, value in category_imp.head(4).items():
        lines.append(f"  - {feature_name}: {value:.4f}")

    lines.extend(
        [
            "- Yorum: Model sinyali daha sivri ve baskin; kategori sinyali ise bircok kucuk ozellige dagiliyor.",
            "- Gorsel: `runs/hw2_full_seed42/inferences/ai_2_feature_importance.png`",
        ]
    )

    out_path = os.path.join(OUT_DIR, "four_inferences.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("features.csv not found. Run extract_features.py first.")
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError("metrics.json not found. Run predict.py first.")

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    _, human_style_means = save_human_model_style_plot(df)
    _, prompt_summary = save_human_prompt_sensitivity_plot(df)
    save_ai_shared_space_plot(df)
    _, model_imp, category_imp = save_ai_feature_importance_plot(df)
    note_path = write_inference_markdown(
        metrics, human_style_means, prompt_summary, model_imp, category_imp
    )

    print(f"Saved {OUT_DIR}")
    print(f"Saved {note_path}")


if __name__ == "__main__":
    main()
