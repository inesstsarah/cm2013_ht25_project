import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import config
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
    Example:
        >>> selected_features, selected_indices = select_features(features, labels, config)
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
        selected_features = features
    
    elif config.CURRENT_ITERATION == 2:
        selected_features = features
        
    elif config.CURRENT_ITERATION == 3:
        selected_features = features
        selected_features, selected_mask = _variance_threshold_selector(features,threshold=0.1)
        selected_indices = selected_indices[selected_mask]
        selected_features, selected_mask = _select_features_correlation(selected_features)
        selected_indices = selected_indices[selected_mask]
        selected_features, selected_mask = _select_features_mutual_information(selected_features, labels, config.FEATURE_SELECTION_K)
        selected_indices = selected_indices[selected_mask]
        
    elif config.CURRENT_ITERATION == 4:
        # Advanced feature selection for Random Forest
        print("Using advanced feature selection optimized for Random Forest...")
        selected_features = features
        
        # Step 1: Remove low variance features
        selected_features, selected_mask = _variance_threshold_selector(selected_features, threshold=0.1)
        selected_indices = selected_indices[selected_mask]
        
        # Step 2: Remove highly correlated features
        selected_features, kept_indices_corr = _select_features_correlation(selected_features)
        # kept_indices_corr is a list of indices relative to current selected_features
        # Map them back to original feature indices
        selected_indices = selected_indices[kept_indices_corr]
        
        # Step 3: Use Random Forest-based feature importance for final selection
        # This is more suitable for Random Forest classifier than mutual information
        selected_features, top_k_indices_rf = _select_features_rf_importance(
            selected_features, labels, config.FEATURE_SELECTION_K
        )
        # top_k_indices_rf are indices relative to current selected_features
        # Map them back to original feature indices using index array indexing
        selected_indices = selected_indices[top_k_indices_rf]

    # Ensure selected_indices is a numpy array and convert to int for indexing
    selected_indices = np.asarray(selected_indices, dtype=int)
    print(f"Selected features shape: {selected_features.shape}")
    print(f"Final selected_indices shape: {selected_indices.shape}, dtype: {selected_indices.dtype}")
    print(f"Final selected_indices range: [{selected_indices.min()}, {selected_indices.max()}]")
    return selected_features, selected_indices


def _variance_threshold_selector(features, threshold=0.0):
    """
    Selects features based on a variance threshold strategy.

    This function operates in two modes:
    1. If threshold <= 0: Removes constant features (variance == 0).
    2. If threshold > 0: Removes the bottom N% of features based on variance ranking,
       where N is determined by (threshold * 100).

    Args:
        features (np.ndarray): The input features array of shape (n_samples, n_features).
        threshold (float): The threshold for selection.
            - If <= 0.0, removes features with 0 variance.
            - If > 0.0, represents a percentile rank (0.0 to 1.0) to exclude.
              For example, 0.1 removes the bottom 10% of features with the lowest variance.

    Returns:
        tuple: A tuple containing:
            - selected_features (np.ndarray): The transformed feature subset of shape (n_samples, n_selected_features).
            - support_mask (np.ndarray): A boolean mask indicating which features were retained.

    Example:
        >>> # Remove bottom 20% of features by variance
        >>> features_subset, mask = _variance_threshold_selector(X, threshold=0.2)
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
    
    Returns:
        np.ndarray: The selected features (n_samples, n_selected_features).

    Example:
        >>> selected_features = _select_features_correlation(features)
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
    
    Example:
        >>> selected_features = _select_features_mutual_information(features, labels, k)
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

def _select_features_rf_importance(features: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Select features using Random Forest feature importance.
    This method is optimized for Random Forest classifiers.
    
    Args:
        features (np.ndarray): The input features (n_samples, n_features).
        labels (np.ndarray): The corresponding labels.
        k (int): The number of top features to select.
    
    Returns:
        tuple: (selected_features, selected_mask)
    """
    k = min(k, features.shape[1])
    print(f"\nUsing Random Forest Feature Importance to select top {k} features...")
    print(f"  Method: RandomForestClassifier with feature_importances_")
    print(f"  Optimized for Random Forest classifier")
    
    # Train a Random Forest to get feature importance
    # Use a quick RF with reasonable parameters
    rf_selector = RandomForestClassifier(**config.BEST_PARAMS)
    
    # Scale features before training RF (for consistency)
    scaler_temp = StandardScaler()
    features_scaled = scaler_temp.fit_transform(features)
    
    rf_selector.fit(features_scaled, labels)
    
    # Get feature importances
    importances = rf_selector.feature_importances_
    
    # Select top k features based on importance
    top_k_indices = np.argsort(importances)[-k:][::-1]
    
    selected_features = features[:, top_k_indices]
    
    print(f"  Selected {len(top_k_indices)} features from {features.shape[1]} total")
    print(f"  Selected features shape: {selected_features.shape}")
    print(f"  Top 5 feature importances: {sorted(importances, reverse=True)[:5]}\n")
    print(f"  Selected indices (relative to input): {top_k_indices[:10] if len(top_k_indices) > 10 else top_k_indices}...")  # Debug info
    
    # Return indices array (not boolean mask) for consistency with mutual information method
    # This ensures correct indexing when mapping back to original feature space
    return selected_features, top_k_indices