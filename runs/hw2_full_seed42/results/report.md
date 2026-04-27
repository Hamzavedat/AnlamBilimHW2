# Experiment Report

- Rows: 2000

## Four Trajectory Inferences
- Human 1: raw geometric features
- Human 2: normalized and scale-invariant geometric features
- AI-assisted 1: within-model PCA projection features
- AI-assisted 2: combined normalized geometry + PCA projection features

## Model Identity
- human_raw_geometry RF accuracy: 1.0000
- human_raw_geometry LR accuracy: 0.9975
- human_normalized_geometry RF accuracy: 0.9950
- human_normalized_geometry LR accuracy: 0.9975
- ai_pca_projection RF accuracy: 1.0000
- ai_pca_projection LR accuracy: 1.0000
- ai_combined_geometry RF accuracy: 1.0000
- ai_combined_geometry LR accuracy: 0.9975

## Category Prediction (Grouped)
- human_raw_geometry RF accuracy: 0.2667
- human_raw_geometry LR accuracy: 0.1786
- human_normalized_geometry RF accuracy: 0.2381
- human_normalized_geometry LR accuracy: 0.1857
- ai_pca_projection RF accuracy: 0.2048
- ai_pca_projection LR accuracy: 0.1595
- ai_combined_geometry RF accuracy: 0.2476
- ai_combined_geometry LR accuracy: 0.1762
- Coverage satisfied: True