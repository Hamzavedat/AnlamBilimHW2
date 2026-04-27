# Experiment Report

- Rows: 2000

## Four Trajectory Inferences
- Human 1: raw geometric features
- Human 2: normalized and scale-invariant geometric features
- AI-assisted 1: within-model PCA projection features
- AI-assisted 2: combined normalized geometry + PCA projection features

## Model Identity
- human_raw_geometry RF accuracy: 1.0000
- human_raw_geometry LR accuracy: 1.0000
- human_normalized_geometry RF accuracy: 1.0000
- human_normalized_geometry LR accuracy: 1.0000
- ai_pca_projection RF accuracy: 1.0000
- ai_pca_projection LR accuracy: 0.9900
- ai_combined_geometry RF accuracy: 1.0000
- ai_combined_geometry LR accuracy: 1.0000

## Category Prediction (Grouped)
- human_raw_geometry RF accuracy: 0.0725
- human_raw_geometry LR accuracy: 0.0975
- human_normalized_geometry RF accuracy: 0.0925
- human_normalized_geometry LR accuracy: 0.1100
- ai_pca_projection RF accuracy: 0.0900
- ai_pca_projection LR accuracy: 0.0875
- ai_combined_geometry RF accuracy: 0.0800
- ai_combined_geometry LR accuracy: 0.1225
- Coverage satisfied: True