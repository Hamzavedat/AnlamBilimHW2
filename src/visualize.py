import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RUN_NAME = "hw2_full_seed42"
FEATURES_PATH = os.path.join("runs", RUN_NAME, "data", "features.csv")
PLOTS_DIR = os.path.join("runs", RUN_NAME, "plots")
SEED = 42
SAMPLES_PER_MODEL = 3


def normalize_trajectory(trajectory):
    centered = trajectory - trajectory[0]
    diffs = np.diff(centered, axis=0)
    if len(diffs) == 0:
        return centered

    scale = np.mean(np.linalg.norm(diffs, axis=1))
    if scale < 1e-10:
        scale = 1.0
    return centered / scale


def plot_model_panels(df):
    models = sorted(df["model_name"].unique())
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    axes = axes[0]

    for ax, model_name in zip(axes, models):
        model_df = df[df["model_name"] == model_name]
        sample_count = min(SAMPLES_PER_MODEL, len(model_df))
        sample_indices = rng.choice(len(model_df), size=sample_count, replace=False)
        sampled_rows = model_df.iloc[sample_indices]

        trajectories = []
        for _, row in sampled_rows.iterrows():
            traj = np.load(row["trajectory_path"]).astype(np.float64)
            trajectories.append(normalize_trajectory(traj))

        all_points = np.vstack(trajectories)
        pca = PCA(n_components=2, random_state=SEED)
        pca.fit(all_points)

        for i, traj in enumerate(trajectories):
            traj_2d = pca.transform(traj)
            ax.plot(traj_2d[:, 0], traj_2d[:, 1], lw=1.5, alpha=0.8, label=f"traj_{i}")
            ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color="green", s=20)
            ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color="red", marker="x", s=35)

        ax.set_title(f"{model_name}\nmodel-specific PCA")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

    fig.suptitle("Panels use different PCA bases, so axes are not directly comparable", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "trajectories_model_specific_pca.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_common_space(df):
    feature_cols = [col for col in df.columns if col.startswith("norm_geo_") or col.startswith("si_")]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    X_scaled = StandardScaler().fit_transform(X)
    X_2d = PCA(n_components=2, random_state=SEED).fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    for model_name in sorted(df["model_name"].unique()):
        mask = df["model_name"] == model_name
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=18, alpha=0.7, label=model_name)

    ax.set_title("Common feature-space PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend()

    out_path = os.path.join(PLOTS_DIR, "common_feature_space_pca.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("features.csv not found. Run extract_features.py first.")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    plot_model_panels(df)
    plot_common_space(df)


if __name__ == "__main__":
    main()
