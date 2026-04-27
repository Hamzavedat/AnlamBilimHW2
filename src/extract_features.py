import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm


RUN_NAME = "hw2_full_seed42"
RUN_DIR = os.path.join("runs", RUN_NAME)
TABLES_DIR = os.path.join(RUN_DIR, "tables")
FEATURES_PATH = os.path.join(TABLES_DIR, "features.csv")
SEED = 42
PCA_FIT_SAMPLES = 50
EPS = 1e-10


def calculate_geometric_features(trajectory, prefix):
    diffs = np.diff(trajectory, axis=0)
    if len(diffs) == 0:
        return {
            f"{prefix}_dist_mean": 0.0,
            f"{prefix}_dist_std": 0.0,
            f"{prefix}_dist_max": 0.0,
            f"{prefix}_dist_min": 0.0,
            f"{prefix}_cos_sim_mean": 1.0,
            f"{prefix}_cos_sim_std": 0.0,
            f"{prefix}_start_end_dist": 0.0,
            f"{prefix}_path_length": 0.0,
            f"{prefix}_linearity": 0.0,
        }

    distances = np.linalg.norm(diffs, axis=1)

    if len(diffs) > 1:
        dot_products = np.sum(diffs[:-1] * diffs[1:], axis=1)
        norms = np.linalg.norm(diffs[:-1], axis=1) * np.linalg.norm(diffs[1:], axis=1)
        norms = np.where(norms == 0, EPS, norms)
        cos_sims = dot_products / norms
    else:
        cos_sims = np.array([1.0], dtype=np.float64)

    start_end_dist = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    path_len = float(np.sum(distances))

    return {
        f"{prefix}_dist_mean": float(np.mean(distances)),
        f"{prefix}_dist_std": float(np.std(distances)),
        f"{prefix}_dist_max": float(np.max(distances)),
        f"{prefix}_dist_min": float(np.min(distances)),
        f"{prefix}_cos_sim_mean": float(np.mean(cos_sims)),
        f"{prefix}_cos_sim_std": float(np.std(cos_sims)),
        f"{prefix}_start_end_dist": start_end_dist,
        f"{prefix}_path_length": path_len,
        f"{prefix}_linearity": float(start_end_dist / (path_len + EPS)),
    }


def normalize_trajectory(trajectory):
    centered = trajectory - trajectory[0]
    diffs = np.diff(centered, axis=0)
    if len(diffs) == 0:
        return centered, 1.0

    step_norms = np.linalg.norm(diffs, axis=1)
    scale = float(np.mean(step_norms))
    if scale < EPS:
        scale = 1.0
    return centered / scale, scale


def calculate_scale_invariant_features(trajectory):
    diffs = np.diff(trajectory, axis=0)
    if len(diffs) == 0:
        return {
            "si_step_cv": 0.0,
            "si_step_max_over_mean": 0.0,
            "si_step_min_over_mean": 0.0,
            "si_turn_mean": 1.0,
            "si_turn_std": 0.0,
            "si_turn_abs_mean": 1.0,
            "si_linearity": 0.0,
        }

    distances = np.linalg.norm(diffs, axis=1)
    dist_mean = float(np.mean(distances))
    dist_std = float(np.std(distances))
    dist_max = float(np.max(distances))
    dist_min = float(np.min(distances))

    if len(diffs) > 1:
        dot_products = np.sum(diffs[:-1] * diffs[1:], axis=1)
        norms = np.linalg.norm(diffs[:-1], axis=1) * np.linalg.norm(diffs[1:], axis=1)
        norms = np.where(norms == 0, EPS, norms)
        cos_sims = dot_products / norms
    else:
        cos_sims = np.array([1.0], dtype=np.float64)

    start_end_dist = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
    path_len = float(np.sum(distances))

    return {
        "si_step_cv": float(dist_std / (dist_mean + EPS)),
        "si_step_max_over_mean": float(dist_max / (dist_mean + EPS)),
        "si_step_min_over_mean": float(dist_min / (dist_mean + EPS)),
        "si_turn_mean": float(np.mean(cos_sims)),
        "si_turn_std": float(np.std(cos_sims)),
        "si_turn_abs_mean": float(np.mean(np.abs(cos_sims))),
        "si_linearity": float(start_end_dist / (path_len + EPS)),
    }


def calculate_2d_features(trajectory, reducer):
    if len(trajectory) < 3:
        return {
            "within_model_pca_area": 0.0,
            "within_model_pca_spread": 0.0,
            "within_model_pca_linearity": 1.0,
        }

    traj_2d = reducer.transform(trajectory)
    mins = np.min(traj_2d, axis=0)
    maxs = np.max(traj_2d, axis=0)
    area = float(np.prod(maxs - mins))
    spread = float(np.sum(np.var(traj_2d, axis=0)))

    diffs = np.diff(traj_2d, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    path_len = float(np.sum(distances))
    start_end_dist = float(np.linalg.norm(traj_2d[-1] - traj_2d[0]))

    return {
        "within_model_pca_area": area,
        "within_model_pca_spread": spread,
        "within_model_pca_linearity": float(start_end_dist / (path_len + EPS)),
    }


def fit_pca_models(metadata):
    rng = np.random.default_rng(SEED)
    pca_models = {}

    for model_name in metadata["model_name"].unique():
        model_rows = metadata[metadata["model_name"] == model_name]
        sample_count = min(PCA_FIT_SAMPLES, len(model_rows))
        sample_indices = rng.choice(len(model_rows), size=sample_count, replace=False)

        samples = []
        for idx in sample_indices:
            traj = np.load(model_rows.iloc[idx]["trajectory_path"]).astype(np.float64)
            norm_traj, _ = normalize_trajectory(traj)
            samples.append(norm_traj)

        stacked = np.vstack(samples)
        pca = PCA(n_components=2, random_state=SEED)
        pca.fit(stacked)
        pca_models[model_name] = pca

    return pca_models


def main():
    metadata_files = glob(os.path.join(TABLES_DIR, "*_metadata.csv"))
    if not metadata_files:
        raise FileNotFoundError("Metadata files not found. Run collect_data.py first.")

    metadata = pd.concat([pd.read_csv(path) for path in metadata_files], ignore_index=True)
    print(f"Loaded metadata for {len(metadata)} trajectories.")

    pca_models = fit_pca_models(metadata)
    rows = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting features"):
        traj = np.load(row["trajectory_path"]).astype(np.float64)
        norm_traj, scale = normalize_trajectory(traj)

        feature_row = {"traj_id": row["traj_id"], "trajectory_scale": float(scale)}
        feature_row.update(calculate_geometric_features(traj, "raw_geo"))
        feature_row.update(calculate_geometric_features(norm_traj, "norm_geo"))
        feature_row.update(calculate_scale_invariant_features(traj))
        feature_row.update(calculate_2d_features(norm_traj, pca_models[row["model_name"]]))
        rows.append(feature_row)

    feature_df = pd.DataFrame(rows)
    final_df = metadata.merge(feature_df, on="traj_id")

    if os.path.exists(FEATURES_PATH):
        os.remove(FEATURES_PATH)
    final_df.to_csv(FEATURES_PATH, index=False)
    print(f"Saved {FEATURES_PATH}")


if __name__ == "__main__":
    main()
