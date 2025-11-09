import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_features(features, labels, config):
    """
    STUDENT IMPLEMENTATION AREA: Select most relevant features.

    Feature selection becomes important in later iterations to:
    1. Reduce overfitting
    2. Improve computation speed
    3. Focus on most discriminative features
    4. Handle curse of dimensionality

    Suggested approaches for students to implement:
    - Statistical tests (ANOVA F-test, chi-square)
    - Mutual information
    - Correlation-based selection
    - Recursive feature elimination
    - L1 regularization (LASSO)
    - Tree-based feature importance

    Args:
        features (np.ndarray): The input features (n_samples, n_features).
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.

    Returns:
        np.ndarray: The selected features (n_samples, n_selected_features).
    """
    print(f"Selecting features for iteration {config.CURRENT_ITERATION}...")
    print(f"Input features shape: {features.shape}")

    if features.shape[1] == 0:
        print("⚠️  WARNING: No features to select from!")
        return features

    if config.CURRENT_ITERATION <= 2:
        selected_features = features
        # selected_features = _select_features_mutual_information(features, labels, config.FEATURE_SELECTION_K)
        
    elif config.CURRENT_ITERATION == 3:
        selected_features = features
        
    elif config.CURRENT_ITERATION == 4:
        # TODO: Students should implement advanced feature selection
        print("TODO: Students should implement advanced feature selection for iteration 4")
        print("Suggested: Use more sophisticated methods like RFE or feature importance")

        # Placeholder - students must replace:
        selected_features = features  # No selection implemented yet

    #print(f"Selected features shape: {selected_features.shape}")
    return selected_features


def _select_features_mutual_information(features: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Select features using mutual information.
    Args:
        features (np.ndarray): The input features (n_samples, n_features).
        labels (np.ndarray): The corresponding labels.
        k (int): The number of top features to select.
    Returns:
        np.ndarray: The selected features (n_samples, n_selected_features).
    """
    k = min(k, features.shape[1])
    print(f"Using Mutual Information to select top {k} features...")
    print(f"  Method: mutual_info_classif (Option B)")
    print(f"  Captures both linear and non-linear relationships")

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selected_features = selector.fit_transform(features, labels)

    feature_scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    print(f"  Selected {len(selected_indices)} features from {features.shape[1]} total")
    print(f"  Selected features shape: {selected_features.shape}")
    print(f"  Top 5 feature scores: {sorted(feature_scores, reverse=True)[:5]}")
    return selected_features