import numpy as np
import os
import sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import preprocess, highpass_filter, notch_filter, bandpass_filter
import config
from src.data_loader import load_training_data
import matplotlib.pyplot as plt
from src.visualization import visualize_fft, visualize_signal


def test_notch_filter_edf():

    edf_file = "../data/training/R1.edf"
    xml_file = "../data/training/R1.xml"
    multi_channel_data, labels, channel_info = load_training_data(
                edf_file, xml_file
            )
    
    eeg_data = multi_channel_data['eeg']
    eeg_fs = channel_info['eeg_fs'] 
    print(f"EEG sampling frequency: {eeg_fs} Hz")
    preprocessed_eeg = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[1]):
        print(f"Processing EEG channel {ch+1}")
        nepochs = eeg_data.shape[0]
        signal = eeg_data[:, ch, :].flatten()
        # Apply EEG-specific preprocessing
        filtered_signal = notch_filter(signal, config.NOTCH_FILTER_FREQ, config.NOTCH_FILTER_Q, eeg_fs)
        assert isinstance(filtered_signal, np.ndarray)
        assert filtered_signal.shape == signal.shape


def test_bandpass_filter_edf():

    edf_file = "../data/training/R1.edf"
    xml_file = "../data/training/R1.xml"
    multi_channel_data, labels, channel_info = load_training_data(
                edf_file, xml_file
            )
    
    eeg_data = multi_channel_data['eeg']
    eeg_fs = channel_info['eeg_fs'] 
    print(f"EEG sampling frequency: {eeg_fs} Hz")
    preprocessed_eeg = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[1]):
        print(f"Processing EEG channel {ch+1}")
        nepochs = eeg_data.shape[0]
        signal = eeg_data[:, ch, :].flatten()
        # Apply EEG-specific preprocessing
        filtered_signal = bandpass_filter(signal, config.BANDPASS_FILTER_LOWER_FREQ, config.BANDPASS_FILTER_HIGHER_FREQ, eeg_fs, config.BANDPASS_FILTER_ORDER)
        assert isinstance(filtered_signal, np.ndarray)
        assert filtered_signal.shape == signal.shape

  

def test_highpass_filter():
    """Function to test highpass filter for baseline wander removal"""

    fs = 100
    cutoff = 10
    data = np.sin(2 * np.pi * 5 * np.arange(0, 10, 1/fs)) + np.sin(2 * np.pi * 20 * np.arange(0, 10, 1/fs))
    filtered_data = highpass_filter(data, cutoff, fs)
    assert isinstance(filtered_data, np.ndarray)
    assert filtered_data.shape == data.shape
    # Basic check: ensure some attenuation of high frequency component
    # This is a very simple check and might need more sophisticated validation
    assert np.std(filtered_data) < np.std(data) # Expect some reduction in signal power


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])