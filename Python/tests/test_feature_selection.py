"""
Tests for the feature selection module
This can be used to test the functions in the feature selection module 
"""
import pytest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

from src.feature_selection import (
    _select_features_mutual_information
)

from src.feature_selection import (
    _select_features_mutual_information,
    variance_threshold_selector
)

def test_select_features_variance_threshold():
    np.random.seed(42)
    features = np.random.rand(100, 10)
    features[:, 0] = 0.5  # Zero variance

    selected_features = variance_threshold_selector(features, threshold=0.1)
    assert selected_features.shape[1] == 9  # Should remove 1 low-variance feature

def test_select_features_mutual_information():
    features = np.random.rand(1000, 100)
    labels = np.random.randint(0, 2, 1000)
    selected_features = _select_features_mutual_information(features, labels, config.FEATURE_SELECTION_K)
    assert selected_features.shape[1] == config.FEATURE_SELECTION_K
    print(selected_features)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])