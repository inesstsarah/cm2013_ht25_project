from scipy.signal import butter, lfilter, iirnotch, filtfilt, welch
import numpy as np
import matplotlib.pyplot as plt
from src.visualization import visualize_fft

def lowpass_filter(data, cutoff, fs, order=5):
    """
    EXAMPLE IMPLEMENTATION: Simple low-pass Butterworth filter.

    Students should understand this basic filter and consider:
    - Is 40Hz the right cutoff for EEG?
    - What about high-pass filtering?
    - Should you use bandpass instead?
    - What about notch filtering for powerline interference?

    Args:
        data (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    # TODO: Students may want to implement additional filtering:
    # - High-pass filter to remove DC drift
    # - Notch filter for 50/60 Hz powerline noise
    # - Bandpass filter (e.g., 0.5-40 Hz for EEG)

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def notch_filter(data, f0, Q, fs):
    b, a = iirnotch(f0, Q, fs)        
    y = filtfilt(b,a,data) 
    return y

def bandpass_filter(data, lowcut , highcut, fs, order):
    nyquist = 0.5 * fs
    normal_lowcut = lowcut / nyquist
    normal_highcut = highcut/nyquist
    b, a = butter(order, [normal_lowcut, normal_highcut], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y


def preprocess(data, config):
    """
    STUDENT IMPLEMENTATION AREA: Preprocess data based on current iteration.

    This function should handle both single-channel and multi-channel data
    (2 EEG + 2 EOG + 1 EMG channels) based on the data structure.

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        config (module): The configuration module.

    Returns:
        Same format as input: preprocessed data.
    """
    print(f"Preprocessing data for iteration {config.CURRENT_ITERATION}...")

    # Detect data format
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return preprocess_multi_channel(data, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return preprocess_multi_channel(data, config)

def preprocess_multi_channel(multi_channel_data, config):
    """
    Preprocess multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.
    Each channel type may have different sampling rates and require different processing.
    """
    preprocessed_data = {}
    preprocessed_data['eeg'] = preprocess_eeg_channel(multi_channel_data['eeg'], config)

    if config.CURRENT_ITERATION >= 2:  # EOG starts in iteration 2
        # Process EOG channels (2 channels) - may need different filtering
        eog_data = multi_channel_data['eog']
        eog_fs = 50  # Actual sampling rate: 50 Hz (TODO: Get from channel_info)
        preprocessed_eog = np.zeros_like(eog_data)

        for ch in range(eog_data.shape[1]):
            for epoch in range(eog_data.shape[0]):
                signal = eog_data[epoch, ch, :]
                # EOG may need different filter settings (preserve slow eye movements)
                filtered_signal = lowpass_filter(signal, 30, eog_fs)  # Lower cutoff for EOG
                preprocessed_eog[epoch, ch, :] = filtered_signal

        preprocessed_data['eog'] = preprocessed_eog

    if config.CURRENT_ITERATION >= 3:  # EMG starts in iteration 3
        # Process EMG channel (1 channel) - may need higher frequency preservation
        emg_data = multi_channel_data['emg']
        emg_fs = 125  # Actual sampling rate: 125 Hz (TODO: Get from channel_info)
        preprocessed_emg = np.zeros_like(emg_data)

        for epoch in range(emg_data.shape[0]):
            signal = emg_data[epoch, 0, :]
            # EMG needs higher frequency content preserved (muscle activity)
            filtered_signal = lowpass_filter(signal, 70, emg_fs)  # Higher cutoff for EMG
            preprocessed_emg[epoch, 0, :] = filtered_signal

        preprocessed_data['emg'] = preprocessed_emg
        print("Multi-channel preprocessing applied to EEG + EOG + EMG")
    elif config.CURRENT_ITERATION >= 2:
        print("Iteration 2: Processing EEG + EOG channels")
    else:
        print("Iteration 1: Processing EEG channels only")

    return preprocessed_data


def preprocess_single_channel(data, config):
    """
    Backward compatibility for single-channel preprocessing.
    """
    if config.CURRENT_ITERATION == 1:
        preprocessed_data = preprocess_eeg_channel(data, config)

    elif config.CURRENT_ITERATION == 2:
        print("TODO: Implement enhanced preprocessing for iteration 2")
        preprocessed_data = data  # Placeholder

    elif config.CURRENT_ITERATION >= 3:
        print("TODO: Students should use multi-channel data format for iteration 3+")
        preprocessed_data = data  # Placeholder

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return preprocessed_data


def preprocess_eeg_channel(eeg_data, config):
    """
    Preprocess single EEG channel data.
    """
    # Process EEG channels (2 channels)
    eeg_fs = 125
    preprocessed_eeg = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[1]):
        print(f"Processing EEG channel {ch+1}")
        nepochs = eeg_data.shape[0]
        signal = eeg_data[:, ch, :].flatten()
        # Apply EEG-specific preprocessing
        filtered_signal = notch_filter(signal, config.NOTCH_FILTER_FREQ, config.NOTCH_FILTER_Q, eeg_fs)
        filtered_signal = bandpass_filter(filtered_signal, config.BANDPASS_FILTER_LOWER_FREQ, config.BANDPASS_FILTER_HIGHER_FREQ, eeg_fs, config.BANDPASS_FILTER_ORDER)
        
        filtered_signal = signal
        # FFT visualization
        plt.figure(figsize=(12, 6))
        fig, axes = plt.subplot(2, 2, 1)
        visualize_fft(signal, eeg_fs, ax=axes[0], title=f"EEG Channel {ch+1} - Original Signal FFT")
        visualize_fft(filtered_signal, eeg_fs, ax=axes[1], title=f"EEG Channel {ch+1} - Filtered Signal FFT")
        preprocessed_eeg[:, ch, :] = filtered_signal.reshape(nepochs, -1)

    return preprocessed_eeg
