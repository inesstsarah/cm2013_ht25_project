"""
Tests for the feature extraction module
This can be used to test the functions in the classification module 
"""
import pytest
import os
import sys
import numpy as np
import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.feature_extraction import (
    hjorth_activity,
    hjorth_mobility,
    hjorth_complexity,
    simple_k_complex_extraction,
    extract_time_domain_features,
    extract_single_channel_features,
    extract_multi_channel_features,
    extract_features
)
from src.data_loader import load_training_data
from src.preprocessing import preprocess


edf_path = os.path.join('../data/sample', 'R1.edf')
xml_path = os.path.join('../data/sample', 'R1.xml')
data,_,channel_info= load_training_data(edf_path, xml_path)
preprocessed_data = preprocess(data,channel_info, config)
epoch_eeg = preprocessed_data['eeg'][0,0,:]

def test_hjorth_activity():
    """Test Hjorth Activity (Variance)"""

    expected_variance = np.var(epoch_eeg)
    activity = hjorth_activity(epoch_eeg)
    assert np.isclose(activity, expected_variance)
    assert isinstance(activity, float)

def test_hjorth_mobility():
    """Test Hjorth Mobility (ratio of std of diff to std of signal)"""
   
    std_diff = np.std(np.diff(epoch_eeg))
    std_epoch = np.std(epoch_eeg)
    expected_mobility = std_diff / std_epoch
    mobility = hjorth_mobility(epoch_eeg)
    assert np.isclose(mobility, expected_mobility)
    assert isinstance(mobility, float)

def test_hjorth_complexity():
    """Test Hjorth Complexity (ratio of Mobility of diff to Mobility of signal)"""
 
    mobility_diff = hjorth_mobility(np.diff(epoch_eeg))
    mobility_epoch = hjorth_mobility(epoch_eeg)
    expected_complexity = mobility_diff / mobility_epoch
    complexity = hjorth_complexity(epoch_eeg)
    assert np.isclose(complexity, expected_complexity)
    assert isinstance(complexity, float)


def test_simple_k_complex_extraction_output_type():
    """Test K-complex extraction returns correct types and basic values"""
 
    nb, duration = simple_k_complex_extraction(epoch_eeg)
    assert isinstance(nb, int)
    assert isinstance(duration, float)
    assert nb >= 0
    assert duration >= 0


def test_extract_time_domain_features():
    """Test that extract_time_domain_features returns the correct number of features (18)"""
 
    features = extract_time_domain_features(epoch_eeg)
    # The function extracts 16 explicitly listed features + 2 K-complex features = 18 total
    expected_count = 18
    assert len(features) == expected_count
    assert isinstance(features, dict)
    float_features = [
    'mean', 'median', 'std', 'variance', 'rms', 'min', 'max', 'range', 
    'skewness', 'kurtosis', 'hjorth_activity', 'hjorth_mobility', 
    'hjorth_complexity', 'total_energy', 'mean_power', 'duration_complexes'
    ]

    # Zero crossings and number of K-complexes should be integers.
    int_features = [
        'zero_crossings', 'nb_complexes'
    ]

    # --- Assertions for Statistical Features ---
    for key in float_features:
        assert key in features
        assert isinstance(features[key], float)

    # --- Assertions for Integer/Count Features ---
    for key in int_features:
        assert key in features
        assert isinstance(features[key], (int, np.integer))


def test_extract_single_channel_features_iter1():
    """Test single-channel feature extraction for Iteration 1"""
    single_data = preprocessed_data['eeg'][:,0,:]
    features = extract_single_channel_features(single_data, config)
    assert isinstance(features, np.ndarray)
    # Should route to extract_single_channel_features (1083 epochs, 18 features)
    assert features.shape == (1083, 18)

""" def test_extract_single_channel_features_iter2_and_3():
    Test single-channel feature extraction for Iterations 2 and 3
"""

def test_extract_multi_channel_features_iter1():
    """Test multi-channel feature extraction for Iteration 1 (EEG only)"""
  
    features = extract_multi_channel_features(preprocessed_data, config)
    
    # 2 EEG channels * 18 features/channel
    expected_n_features = 2 * 18
    expected_n_epochs = 1083
    assert isinstance(features, np.ndarray)
    assert features.shape == (expected_n_epochs, expected_n_features)
    
"""def test_extract_multi_channel_features_iter3():
    Test multi-channel feature extraction for Iteration 3 (EEG + EOG + EMG)"""

def test_extract_features_router():
    """Test the main extract_features function routes correctly"""
    # Single-channel routing
    single_data = preprocessed_data['eeg'][:,0,:]
    single_features = extract_features(single_data, config)
    # Should route to extract_single_channel_features (1083 epochs, 18 features)
    assert single_features.shape == (1083, 18)

    # Multi-channel routing
    multi_data = preprocessed_data
    multi_features = extract_features(multi_data, config)
    # Should route to extract_multi_channel_features (1083 epochs, 2 EEG * 18 = 36 features)
    assert multi_features.shape == (1083, 36)

if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])