"""
Tests for the feature extraction module
This can be used to test the functions in the classification module 
"""
import pytest
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.feature_extraction import (
    extract_hjorth_activity,
    extract_hjorth_mobility,
    extract_hjorth_complexity,
    extract_time_domain_features,
    extract_single_channel_features,
    extract_multi_channel_features,
    extract_features,
    welch_method,
    extract_welch_features,
    wavelet_processing,
    wavelet_decomposition,
    wavelet_feature_extraction,
    # AR features
    extract_ar_features,
    _integrate_band_power,
    _peak_frequency,
    _spectral_entropy,
    _extract_derivative_features,
    _spectral_edge_frequency
)
from src.data_loader import load_training_data
from src.preprocessing import preprocess


edf_path = os.path.join('../data/training/', 'R1.edf')
xml_path = os.path.join('../data/training/', 'R1.xml')
data,_,channel_info= load_training_data(edf_path, xml_path)
preprocessed_data = preprocess(data,channel_info, config)
epoch_eeg = preprocessed_data['eeg'][0,0,:]

def test_wavelet_decomposition():
    c = wavelet_decomposition(signal = epoch_eeg, wavelet_name = "coif1")
    # Check if number of elements in array is more than one (there are more than one coefficients)
    assert len(c)>1

def test_wavelet_features():
    """Test Wavelet Decomposition and Feature Extraction"""

    wavelet_features = wavelet_processing(epoch_eeg, "coif1")
    # Check if wavelet features returns a value
    assert len(wavelet_features)>0


def test_hjorth_activity():
    """Test Hjorth Activity (Variance)"""

    expected_variance = np.var(epoch_eeg)
    activity = extract_hjorth_activity(epoch_eeg)
    assert np.isclose(activity, expected_variance)
    assert isinstance(activity, float)

def test_hjorth_mobility():
    """Test Hjorth Mobility (ratio of std of diff to std of signal)"""
   
    std_diff = np.std(np.diff(epoch_eeg))
    std_epoch = np.std(epoch_eeg)
    expected_mobility = std_diff / std_epoch
    mobility = extract_hjorth_mobility(epoch_eeg)
    assert np.isclose(mobility, expected_mobility)
    assert isinstance(mobility, float)

def test_hjorth_complexity():
    """Test Hjorth Complexity (ratio of Mobility of diff to Mobility of signal)"""
 
    mobility_diff = extract_hjorth_mobility(np.diff(epoch_eeg))
    mobility_epoch = extract_hjorth_mobility(epoch_eeg)
    expected_complexity = mobility_diff / mobility_epoch
    complexity = extract_hjorth_complexity(epoch_eeg)
    assert np.isclose(complexity, expected_complexity)
    assert isinstance(complexity, float)

def test_extract_time_domain_features():
    """Test that extract_time_domain_features returns the correct number of features (18)"""
 
    features = extract_time_domain_features(epoch_eeg)
    # The function extracts 16 features
    expected_count = 16
    assert len(features) == expected_count
    assert isinstance(features, dict)
    float_features = [
    'mean', 'median', 'std', 'variance', 'rms', 'min', 'max', 'range', 
    'skewness', 'kurtosis', 'hjorth_activity', 'hjorth_mobility', 
    'hjorth_complexity', 'total_energy', 'entropy'
    ]

    # Zero crossings
    int_features = [
        'zero_crossings'
    ]

    # Assertions for Statistical Features 
    for key in float_features:
        assert key in features
        assert isinstance(features[key], float)

    # Assertions for Integer/Count Features 
    for key in int_features:
        assert key in features
        assert isinstance(features[key], (int, np.integer))


def test_extract_single_channel_features_iter1():
    """Test single-channel feature extraction for Iteration 1"""
    single_data = preprocessed_data['eeg'][0:2,0,:]
    features = extract_single_channel_features(single_data, config)
    assert isinstance(features, np.ndarray)
    # Should route to extract_single_channel_features (1083 epochs, 16 features)
    assert features.shape == (2, 16)

def test_extract_multi_channel_features_iter1():
    """Test multi-channel feature extraction for Iteration 1 (EEG only)"""
  
    features = extract_multi_channel_features(preprocessed_data, config)
    # 2 EEG channels * 16 features/channel
    expected_n_features = 2 * 16
    expected_n_epochs = 1083
    assert isinstance(features, np.ndarray)
    assert features.shape == (expected_n_epochs, expected_n_features)
    
"""def test_extract_multi_channel_features_iter3():
    Test multi-channel feature extraction for Iteration 3 (EEG + EOG + EMG)"""

def test_extract_features_router_iter1():
    """Test the main extract_features function routes correctly"""
    # Single-channel routing
    single_data = preprocessed_data['eeg'][0:2,0,:]
    single_features = extract_features(single_data, channel_info, config)
    # Should route to extract_single_channel_features (1083 epochs, 16 features)
    assert isinstance(single_features, np.ndarray)
    assert single_features.shape == (2, 16)

    # Multi-channel routing
    multi_data = preprocessed_data
    multi_features = extract_features(multi_data,channel_info, config)
    # Should route to extract_multi_channel_features (1083 epochs, 2 EEG * 16 = 32 features)
    assert isinstance(multi_features, np.ndarray)
    assert multi_features.shape == (1083, 32)

def test_welch_method():
    freqs, psd = welch_method(epoch_eeg, channel_info['eeg_fs'], config)
    assert isinstance(freqs, np.ndarray)
    assert isinstance(psd, np.ndarray)
    assert len(freqs) == len(psd)
    assert np.all(freqs >= 0)
    assert np.all(psd >= 0)

def test_extract_welch_features():
    features = extract_welch_features(epoch_eeg,channel_info['eeg_fs'], config)
    assert isinstance(features, dict)
    for band in config.EEG_BANDS.keys():
        assert "welch_" + band + "_power" in features
        assert "welch_" + band + "_power_rel" in features
        assert isinstance(features["welch_" + band + "_power"], float)
        assert isinstance(features["welch_" + band + "_power_rel"], float)
    assert "welch_spectral_entropy" in features
    assert isinstance(features["welch_spectral_entropy"], float)
    assert "welch_peak_freq" in features
    assert isinstance(features["welch_peak_freq"], float)
    assert "welch_sef90" in features
    assert "welch_sef95" in features


# ========== Tests for AR spectral feature functions ==========

def test_integrate_band_power():
    freqs = np.linspace(0, 50, 100)
    psd = np.ones(100)  # Flat PSD
    power = _integrate_band_power(freqs, psd, 5.0, 15.0)
    assert isinstance(power, float)
    assert power >= 0
    # For flat PSD, power should be approximately (15-5) * 1 = 10
    assert np.isclose(power, 10.0, rtol=0.2)


def test_peak_frequency():
    freqs = np.linspace(0, 50, 100)
    psd = np.zeros(100)
    # Create peak at 10 Hz
    peak_idx = int(10 / 50 * 100)
    psd[peak_idx] = 100.0
    peak_freq = _peak_frequency(freqs, psd, 5.0, 15.0)
    assert isinstance(peak_freq, float)
    assert np.isclose(peak_freq, 10.0, rtol=0.2)


def test_spectral_entropy():
    freqs = np.linspace(0, 50, 100)
    psd = np.ones(100)  # Uniform PSD
    entropy = _spectral_entropy(freqs, psd, 0.0, 50.0)
    assert isinstance(entropy, float)
    assert 0 <= entropy <= 1
    # For uniform distribution, normalized entropy should be close to 1
    assert np.isclose(entropy, 1.0, rtol=0.2)


def test_extract_derivative_features():
    freqs = np.linspace(0, 50, 100)
    psd = np.sin(2 * np.pi * freqs / 10) + 1  # Sinusoidal PSD
    features = _extract_derivative_features(freqs, psd, 0.0, 50.0)
    
    # Check all expected keys are present
    expected_keys = [
        'deriv1_mean', 'deriv1_std', 'deriv1_max', 'deriv1_min', 'deriv1_power',
        'deriv2_mean', 'deriv2_std', 'deriv2_max', 'deriv2_min', 'deriv2_power'
    ]
    assert len(features) == len(expected_keys)
    for key in expected_keys:
        assert key in features
        assert isinstance(features[key], (float, np.floating))
        assert np.isfinite(features[key])


def test_spectral_edge_frequency():
    freqs = np.linspace(0, 50, 100)
    psd = np.ones(100)  # Uniform PSD
    sef90 = _spectral_edge_frequency(freqs, psd, 0.0, 50.0, percentile=0.9)
    assert isinstance(sef90, float)
    assert 0 <= sef90 <= 50.0
    # For uniform PSD, 90% edge should be at 90% of frequency range
    assert np.isclose(sef90, 45.0, rtol=0.2)


def test_extract_ar_features():
    bands = config.EEG_BANDS
    features = extract_ar_features(epoch_eeg, channel_info['eeg_fs'], bands, config.AR_ORDER)
    
    # Check that features is a dictionary
    assert isinstance(features, dict)
    assert len(features) > 0
    
    # Check for expected band power features
    for band_name in bands.keys():
        assert f'ar_{band_name}_power' in features
        assert f'ar_{band_name}_rel_power' in features
        assert f'ar_{band_name}_peak_freq' in features
        
        # Check that power values are non-negative
        assert features[f'ar_{band_name}_power'] >= 0
        assert 0 <= features[f'ar_{band_name}_rel_power'] <= 1
        # Peak frequency should be within band or NaN
        peak_freq = features[f'ar_{band_name}_peak_freq']
        assert np.isnan(peak_freq) or (bands[band_name][0] <= peak_freq <= bands[band_name][1])
    
    # Check for global features
    assert 'ar_spectral_edge_freq' in features
    assert 'ar_peak_frequency' in features
    assert 'ar_spectral_entropy' in features
    
    # Check spectral edge frequency is reasonable
    sef = features['ar_spectral_edge_freq']
    assert np.isnan(sef) or (0 <= sef <= 30.0)
    
    # Check spectral entropy is between 0 and 1
    entropy = features['ar_spectral_entropy']
    assert 0 <= entropy <= 1
    
    # Check derivative features are present
    assert 'deriv1_mean' in features
    assert 'deriv1_std' in features
    assert 'deriv2_mean' in features
    assert 'deriv2_std' in features
    
    # Check that all feature values are finite (not inf or nan, except for peak frequencies which can be nan)
    for key, value in features.items():
        if 'peak_freq' in key or 'edge_freq' in key:
            # Peak frequencies can be NaN
            assert np.isnan(value) or np.isfinite(value)
        else:
            # Other features should be finite
            assert np.isfinite(value), f"Feature {key} is not finite: {value}"


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])