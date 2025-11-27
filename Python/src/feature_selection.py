import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

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
    selected_indices = np.arange(features.shape[1])

    if features.shape[1] == 0:
        print("⚠️  WARNING: No features to select from!")
        return features

    if config.CURRENT_ITERATION == 1:
        # Early iterations: Use all available features
        print("Early iteration - using all available features")
    
    elif config.CURRENT_ITERATION == 2:
        selected_features = features
        # selected_features, selected_mask = variance_threshold_selector(features,threshold=0.1)
        # selected_indices = selected_indices[selected_mask]
        # selected_features, selected_mask = _select_features_correlation(selected_features)
        # selected_indices = selected_indices[selected_mask]
        # selected_features, selected_mask = _select_features_mutual_information(selected_features, labels, config.FEATURE_SELECTION_K)
        # selected_indices = selected_indices[selected_mask]
        
    elif config.CURRENT_ITERATION == 3:
        selected_features = features
        selected_features, selected_mask = variance_threshold_selector(features,threshold=0.1)
        selected_indices = selected_indices[selected_mask]
        selected_features, selected_mask = _select_features_correlation(selected_features)
        selected_indices = selected_indices[selected_mask]
        selected_features, selected_mask = _select_features_mutual_information(selected_features, labels, config.FEATURE_SELECTION_K)
        selected_indices = selected_indices[selected_mask]
        
    elif config.CURRENT_ITERATION == 4:
        # TODO: Students should implement advanced feature selection
        print("TODO: Students should implement advanced feature selection for iteration 4")
        print("Suggested: Use more sophisticated methods like RFE or feature importance")

        # Placeholder - students must replace:
        selected_features = features  # No selection implemented yet

    #print(f"Selected features shape: {selected_features.shape}")
    return selected_features, selected_indices


def variance_threshold_selector(features, threshold=0.0):
    """
    Select features based on variance threshold.

    Args:
        features (np.ndarray): The input features (n_samples, n_features).
        threshold (float): The variance threshold.

    Returns:
        np.ndarray: The selected features (n_samples, n_selected_features).

    Example:
        selected_features = variance_threshold_selector(features, threshold=0.1)
    """
    max_variance = np.var(features, axis=0).max()
    min_variance = np.var(features, axis=0).min()
    print(f"Feature variance range: [{min_variance:.4f}, {max_variance:.4f}]")

    if threshold <= 0:
        selector = VarianceThreshold(threshold=0)
        selector.fit(features)
        selected_features = selector.transform(features)
    else:
        print(f"Applying VarianceThreshold with threshold: {threshold}")
        variances = np.var(features, axis=0)
        cutoff = np.percentile(variances, threshold * 100)
        selector = VarianceThreshold(threshold=cutoff)
        selector = selector.fit(features)
        selected_features = selector.transform(features)

    print(f"Selected features shape after VarianceThreshold: {selected_features.shape}")

    support_mask = selector.get_support()

    return selected_features, support_mask

def _select_features_correlation(features: np.ndarray) -> np.ndarray:
    """
    Select features using correlation, and remove features with high 
    correlation (above 0.95).
    Args: 
        features (np.ndarray): The input features (n_samples, n_features).
    """
    df = pd.DataFrame(features)
    corr_matrix = df.corr(method='pearson', min_periods=1).abs()
    # Select upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
   
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    all_indices = set(range(features.shape[1]))
    drop_indices = set(to_drop)
    kept_indices = sorted(list(all_indices - drop_indices))

    df.drop(to_drop, axis=1, inplace=True)
    # Turn df back into numpy arr
    dropped_features_arr = df.to_numpy()

    return dropped_features_arr, kept_indices

def _select_features_mutual_information(features: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, list[int]]:
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
    print(f"\nUsing Mutual Information to select top {k} features...")
    print(f"  Method: mutual_info_classif (Option B)")
    print(f"  Captures both linear and non-linear relationships")

    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selected_features = selector.fit_transform(features, labels)

    feature_scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    print(f"  Selected {len(selected_indices)} features from {features.shape[1]} total")
    print(f"  Selected features shape: {selected_features.shape}")
    print(f"  Top 5 feature scores: {sorted(feature_scores, reverse=True)[:5]}\n")
    return selected_features, selected_indices