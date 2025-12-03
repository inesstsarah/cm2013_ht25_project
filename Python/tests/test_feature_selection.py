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
    _select_features_mutual_information,
    variance_threshold_selector,
    _select_features_correlation,
    _select_features_rf_importance
)
from src.data_loader import load_training_data
from src.preprocessing import preprocess
import config
import pandas as pd
from src.feature_extraction import extract_features


edf_path = "../data/training/R2.edf"
xml_path = "../data/training/R2.xml"
data,_,channel_info= load_training_data(edf_path, xml_path)
preprocessed_data = preprocess(data,channel_info, config)
epoch_eeg = preprocessed_data['eeg'][0,0,:]
all_features = extract_features(preprocessed_data, channel_info, config)



def test_select_features_variance_threshold():
    np.random.seed(42)
    features = np.random.rand(100, 10)
    features[:, 0] = 0.5  # Zero variance

    selected_features, selected_indices = variance_threshold_selector(features, threshold=0.1)
    assert selected_features.shape[1] == 9  # Should remove 1 low-variance feature
    print(selected_features)


def test_select_features_mutual_information():
    features = np.random.rand(1000, 100)
    labels = np.random.randint(0, 2, 1000)
    selected_features, selected_indices = _select_features_mutual_information(features, labels, config.FEATURE_SELECTION_K)
    assert selected_features.shape[1] == config.FEATURE_SELECTION_K
    assert selected_indices.shape[0] == config.FEATURE_SELECTION_K
    print(selected_features)


def test_correlation_analysis():
    """Function to test correlation analysis algorithm"""
    df = pd.DataFrame(all_features)
    df_corr = df.corr(method='pearson', min_periods=1)
    corr_matrix = df.corr(method='pearson', min_periods=1).abs()
    # Select upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    df.drop(to_drop, axis=1, inplace=True)
    
    corr_matrix_dropped = df.corr(method='pearson', min_periods=1).abs()
    # Select upper triangle
    upper_dropped = corr_matrix_dropped.where(np.triu(np.ones(corr_matrix_dropped.shape), k=1).astype(bool))

    to_drop = [column for column in upper_dropped.columns if any(upper_dropped[column] > 0.95)]

    assert len(to_drop) == 0


def test_select_features_rf_importance():
    features = np.random.rand(1000, 100)
    labels = np.random.randint(0, 2, 1000)
    selected_features, selected_indices = _select_features_rf_importance(features, labels, config.FEATURE_SELECTION_K)
    assert selected_features.shape[1] == config.FEATURE_SELECTION_K
    assert selected_indices.shape[0] == config.FEATURE_SELECTION_K
    print(selected_features)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])# File to test feature selection functions
