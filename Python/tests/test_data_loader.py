import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_training_data, load_holdout_data
import pytest

def test_load_training_data():
    # Test multi-channel training data loading
    dummy_edf_path = os.path.join('../data/training', 'R1.edf')
    dummy_xml_path = os.path.join('../data/training', 'R1.xml')
    multi_channel_data, labels, info = load_training_data(dummy_edf_path, dummy_xml_path)

    # Test multi-channel structure
    assert isinstance(multi_channel_data, dict)
    assert 'eeg' in multi_channel_data
    assert 'eog' in multi_channel_data
    assert 'emg' in multi_channel_data

    # Test data shapes for multi-channel format (125 Hz for EEG, 50 Hz for EOG, 125 Hz for EMG)
    assert multi_channel_data['eeg'].shape[1:] == (2, 3750)  # 240 epochs, 2 EEG channels, 3750 samples (125 Hz * 30s)
    assert multi_channel_data['eog'].shape[1:] == (2, 3750)  # 240 epochs, 2 EOG channels, 1500 samples (50 Hz * 30s)
    assert multi_channel_data['emg'].shape[1:] == (1, 3750)  # 240 epochs, 1 EMG channel, 3750 samples (125 Hz * 30s)

    assert isinstance(labels, np.ndarray)

    assert isinstance(info, dict)
    assert 'eeg_names' in info
    assert 'epoch_length' in info
    assert info['epoch_length'] == 30

def test_load_holdout_data():
    # Test multi-channel holdout data loading
    dummy_holdout_path = os.path.join('../data/holdout', 'H1.edf')
    multi_channel_data, info = load_holdout_data(dummy_holdout_path)

    # Test multi-channel structure
    assert isinstance(multi_channel_data, dict)
    assert 'eeg' in multi_channel_data
    assert isinstance(info, dict)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])