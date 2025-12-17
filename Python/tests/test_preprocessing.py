import numpy as np
import pytest
import os
import sys
from scipy.signal import welch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.preprocessing import (
    highpass_filter, 
    lowpass_filter, 
    notch_filter, 
    bandpass_filter, 
    remove_eog_artifacts, 
    remove_emg_artifacts,
    preprocess
)


def get_power_at_freq(signal, fs, target_freq, bandwidth=2):
    """
    Helper to calculate power spectral density at a specific frequency.
    Returns 0 if the target frequency is out of range (e.g. above Nyquist).
    """
    f, Pxx = welch(signal, fs, nperseg=fs)  # 1 sec window
    
    if target_freq > (fs / 2):
        return 0.0

    idx = np.where((f >= target_freq - bandwidth) & (f <= target_freq + bandwidth))
    
    if len(idx[0]) == 0:
        return 0.0
    
    # Since nperseg=fs, df=1Hz, so sum approx equals integral.
    return np.sum(Pxx[idx])

# --- Fixtures ---

@pytest.fixture
def filter_data():
    """Provides standard time and sampling variables for filter tests."""
    fs = 150 
    duration = 4
    n_samples = duration * fs 
    t = np.arange(n_samples) / fs
    return fs, t

@pytest.fixture
def artifact_data():
    """Provides standard variables for artifact removal tests."""
    fs = 150
    n_samples = 1000
    t = np.arange(n_samples) / fs
    return fs, n_samples, t

# --- Filter Tests ---

def test_highpass_filter(filter_data):
    """Test that low frequencies (0.5 Hz) are attenuated."""
    fs, t = filter_data
    freq_remove = 0.2  # Below cutoff of 0.5
    freq_keep = 10     # Well above cutoff
    
    signal = np.sin(2 * np.pi * freq_remove * t) + np.sin(2 * np.pi * freq_keep * t)
    
    cutoff = config.HIGHPASS_FILTER_FREQ
    filtered = highpass_filter(signal, cutoff, fs)
    
    assert filtered.shape == signal.shape
    
    power_remove = get_power_at_freq(filtered, fs, freq_remove)
    power_keep = get_power_at_freq(filtered, fs, freq_keep)
    
    assert power_keep > (power_remove * 2), "Highpass filter failed to attenuate low frequency"

def test_lowpass_filter(filter_data):
    """Test that high frequencies are attenuated."""
    fs, t = filter_data
    freq_keep = 10
    # 70Hz is valid and should be filtered out by 40Hz Lowpass
    freq_remove = 70 
    
    signal = np.sin(2 * np.pi * freq_keep * t) + np.sin(2 * np.pi * freq_remove * t)
    
    cutoff = config.LOW_PASS_FILTER_FREQ
    filtered = lowpass_filter(signal, cutoff, fs)
    
    power_keep = get_power_at_freq(filtered, fs, freq_keep)
    power_remove = get_power_at_freq(filtered, fs, freq_remove)
    
    assert power_keep > (power_remove * 100), "Lowpass filter failed to attenuate high frequency"

def test_notch_filter(filter_data):
    """Test removal of powerline noise (60Hz)."""
    fs, t = filter_data
    freq_signal = 10
    freq_noise = 60 
    
    signal = np.sin(2 * np.pi * freq_signal * t) + np.sin(2 * np.pi * freq_noise * t)
    
    filtered = notch_filter(signal, f0=60, Q=30, fs=fs)
    
    power_signal = get_power_at_freq(filtered, fs, freq_signal)
    power_noise = get_power_at_freq(filtered, fs, freq_noise)
    
    assert power_signal > (power_noise * 100), "Notch filter failed to remove 60Hz noise"

def test_bandpass_filter(filter_data):
    """Test that frequencies outside 0.5-33 Hz are removed."""
    fs, t = filter_data
    # 10 Hz (Keep), 70 Hz (Remove - High), 0.1 Hz (Remove - Low)
    signal = (np.sin(2 * np.pi * 0.1 * t) + 
              np.sin(2 * np.pi * 10 * t) + 
              np.sin(2 * np.pi * 70 * t))
    
    lowcut = config.BANDPASS_FILTER_LOWER_FREQ
    highcut = config.BANDPASS_FILTER_HIGHER_FREQ
    
    filtered = bandpass_filter(signal, lowcut, highcut, fs, order=4)
    
    power_low = get_power_at_freq(filtered, fs, 0.1)
    power_mid = get_power_at_freq(filtered, fs, 10)
    power_high = get_power_at_freq(filtered, fs, 70)
    
    assert power_mid > (power_low * 10), "Bandpass failed to attenuate low freq"
    assert power_mid > (power_high * 10), "Bandpass failed to attenuate high freq"

# --- Artifact Removal Tests ---

def test_remove_eog_artifacts(artifact_data):
    """Test Regression-based EOG removal."""
    fs, n_samples, t = artifact_data
    n_epochs = 5
    n_channels = 2
    
    np.random.seed(42)
    pure_eeg = np.random.normal(0, 1, (n_epochs, n_channels, n_samples))
    
    # EOG signal 
    eog_signal = 10 * np.sin(2 * np.pi * 2 * t) 
    eog_data = np.tile(eog_signal, (n_epochs, n_channels, 1))
    
    # Contaminate EEG
    contamination_factor = 0.8
    contaminated_eeg = pure_eeg + (contamination_factor * eog_data)
    
    cleaned_eeg = remove_eog_artifacts(contaminated_eeg, eog_data)
    
    original_mse = np.mean((contaminated_eeg - pure_eeg)**2)
    cleaned_mse = np.mean((cleaned_eeg - pure_eeg)**2)
    
    assert cleaned_mse < original_mse, "EOG removal did not improve signal quality"
    assert cleaned_mse < 0.5, "EOG removal failed to recover original signal"

def test_remove_emg_artifacts(artifact_data):
    """Test that EMG artifact remover applies a lowpass filter ONLY when EMG power exceeds the threshold."""
    fs, n_samples, t = artifact_data
    
    # Epoch 0 & 1: Quiet EMG - Create as 3D (1, 1, n_samples)
    epoch_quiet = np.random.normal(0, 1, (1, 1, n_samples)) 
    
    # Epoch 2: High amplitude EMG noise - Create as 3D (1, 1, n_samples)
    epoch_noisy = np.random.normal(0, 100, (1, 1, n_samples)) 
    
    emg_data = np.concatenate([epoch_quiet, epoch_quiet, epoch_noisy], axis=0)
    
    # Create EEG with 60 Hz signal (Valid in 150 Hz fs)
    eeg_signal = np.sin(2 * np.pi * 60 * t)
    eeg_row = eeg_signal.reshape(1, n_samples)
    eeg_data = np.tile(eeg_row, (3, 1, 1)) 
    
    cleaned_eeg = remove_emg_artifacts(eeg_data, emg_data, fs)
    
    # Quiet Epoch (0) should NOT be filtered (60Hz remains)
    power_quiet = get_power_at_freq(cleaned_eeg[0,0,:], fs, 60)
    
    # Noisy Epoch (2) SHOULD be filtered (60Hz removed by 20Hz lowpass in logic)
    power_noisy = get_power_at_freq(cleaned_eeg[2,0,:], fs, 60)
    
    assert power_quiet > 0.1, "Low EMG epoch was incorrectly filtered (Signal lost)"
    assert power_noisy < 0.01, "High EMG epoch was not filtered (Noise remained)"

# --- Pipeline Tests ---

def test_preprocess_multichannel_structure():
    """Test that the dictionary structure is maintained through preprocessing."""
    config.CURRENT_ITERATION = 3
    
    n_epochs, n_samples = 2, 100
    data = {
        'eeg': np.random.rand(n_epochs, 2, n_samples),
        'eog': np.random.rand(n_epochs, 2, n_samples),
        'emg': np.random.rand(n_epochs, 1, n_samples)
    }
    
    channel_info = {
        'eeg_fs': 150,
        'eog_fs': 150,
        'emg_fs': 150
    }
    
    processed = preprocess(data, channel_info, config)
    
    assert isinstance(processed, dict)
    assert 'eeg' in processed
    assert processed['eeg'].shape == data['eeg'].shape

if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])