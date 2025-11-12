from scipy.signal import butter, iirnotch, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt
from src.visualization import visualize_fft, visualize_signal


def highpass_filter(signal,cutoff,fs,order=5):
    """
    Butterworth highpass filter for baseline wander removal.

    Args:
        signal (np.ndarray): The input signal.    
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter. 
    Returns:
        filtered_signal (np.ndarray): The filtered signal.
    
    Example:
        >>> filtered_signal = highpass_filter(signal, cutoff, fs, order)
    """

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal, padlen = 3*order, padtype="odd")
    
    return filtered_signal

def notch_filter(signal, f0, Q, fs):
    """
    IIR notch filter to remove powerline noise.

    Args:
        filter (np.ndarray): The input signal.
        f0 (float): The target frequency to remove.
        Q (float): Quality factor.
        fs (int): The sampling frequency of the signal.
        
    Returns:
        filtered_signal (np.ndarray): The filtered signal.

    Example:
        >>> filtered_signal = notch_filter(signal, f0, Q, fs)
    """

    b, a = iirnotch(f0, Q, fs)        
    filtered_signal = filtfilt(b,a,signal, padlen = 15, padtype="odd")

    return filtered_signal

def bandpass_filter(signal, lowcut , highcut, fs, order):
    """
    Butterworth bandpass filter to retain frequencies within a specific range.
    
    Args:
        signal (np.ndarray): The input signal.
        lowcut (float): The lower cutoff frequency.
        highcut (float): The higher cutoff frequency.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        filtered_signal (np.ndarray): The filtered signal.

    Example:
        >>> filtered_signal = bandpass_filter(signal, lowcut, highcut, fs, order)
    """

    nyquist = 0.5 * fs
    normal_lowcut = lowcut / nyquist
    normal_highcut = highcut/nyquist
    b, a = butter(order, [normal_lowcut, normal_highcut], btype='band', analog=False)
    filtered_signal = filtfilt(b, a, signal, padlen = 3*order, padtype="odd")

    return filtered_signal


def preprocess(data, channel_info, config):
    """
    Preprocess input data based on current iteration settings.

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        config (module): The configuration module.

    Returns:
        Same format as input: preprocessed data.

    Example:
        >>> preprocessed_data = preprocess(data, channel_info, config)
    """

    print(f"Preprocessing data for iteration {config.CURRENT_ITERATION}...")

    # Detect data format
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return preprocess_multi_channel(data,channel_info, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return preprocess_single_channel(data,channel_info, config)


def preprocess_multi_channel(multi_channel_data,channel_info, config):
    """
    Preprocess multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.

    Args:        
        multi_channel_data (dict): Dictionary with keys 'eeg', 'eog', 'emg'.
        config (module): The configuration module.

    Returns:
        dict: Preprocessed multi-channel data with same keys.
    
    Example:
        >>> preprocessed_data = preprocess_multi_channel(multi_channel_data, channel_info, config)
    """

    preprocessed_data = {}
    preprocessed_data['eeg'] = preprocess_eeg_channel(multi_channel_data['eeg'],channel_info, config)

    if config.CURRENT_ITERATION >= 2:  # EOG starts in iteration 2
        # Process EOG channels (2 channels) - may need different filtering
        eog_data = multi_channel_data['eog']
        eog_fs = channel_info['eog_fs']  # Actual sampling rate: 50 Hz (TODO: Get from channel_info)
        preprocessed_eog = np.zeros_like(eog_data)

        for ch in range(eog_data.shape[1]):
            for epoch in range(eog_data.shape[0]):
                signal = eog_data[epoch, ch, :]
                # EOG may need different filter settings (preserve slow eye movements)
                #filtered_signal = lowpass_filter(signal, 30, eog_fs)  # Lower cutoff for EOG
                #preprocessed_eog[epoch, ch, :] = filtered_signal

        preprocessed_data['eog'] = preprocessed_eog

    if config.CURRENT_ITERATION >= 3:  # EMG starts in iteration 3
        # Process EMG channel (1 channel) - may need higher frequency preservation
        emg_data = multi_channel_data['emg']
        emg_fs = 125  # Actual sampling rate: 125 Hz (TODO: Get from channel_info)
        preprocessed_emg = np.zeros_like(emg_data)

        for epoch in range(emg_data.shape[0]):
            signal = emg_data[epoch, 0, :]
            # EMG needs higher frequency content preserved (muscle activity)
            #filtered_signal = lowpass_filter(signal, 70, emg_fs)  # Higher cutoff for EMG
            #preprocessed_emg[epoch, 0, :] = filtered_signal

        preprocessed_data['emg'] = preprocessed_emg
        print("Multi-channel preprocessing applied to EEG + EOG + EMG")
    elif config.CURRENT_ITERATION >= 2:
        print("Iteration 2: Processing EEG + EOG channels")
    else:
        print("Iteration 1: Processing EEG channels only")

    return preprocessed_data


def preprocess_single_channel(data, channel_info, config):
    """
    Backward compatibility for single-channel preprocessing.

    Args:
        data (np.ndarray): A 2D array of shape (n_epochs, n_samples).
        config (module): The configuration module.    

    Returns:
        np.ndarray: A 2D array of preprocessed data (n_epochs, n_samples
    
    Example:
        >>> preprocessed_data = preprocess_single_channel(single_channel_data, channel_info, config)
    """

    if config.CURRENT_ITERATION == 1:
        preprocessed_data = preprocess_eeg_channel(data, channel_info, config)

    elif config.CURRENT_ITERATION == 2:
        print("TODO: Implement enhanced preprocessing for iteration 2")
        preprocessed_data = data  # Placeholder

    elif config.CURRENT_ITERATION >= 3:
        print("TODO: Students should use multi-channel data format for iteration 3+")
        preprocessed_data = data  # Placeholder

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return preprocessed_data


def preprocess_eeg_channel(eeg_data,channel_info, config):
    """
    Preprocess single EEG channel data.
    Args:
        eeg_data (np.ndarray): A 2D array of shape (n_epochs, n_samples, n_channels).
        config (module): The configuration module.

    Returns:
        np.ndarray: Preprocessed EEG data of same shape.

    Example:
        >>> preprocessed_eeg = preprocess_eeg_channel(eeg_data, channel_info, config)
    """

    # Process EEG channels (2 channels)
    eeg_fs = channel_info['eeg_fs'] 
    print(f"EEG sampling frequency: {eeg_fs} Hz")
    preprocessed_eeg = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[1]):
        print(f"Processing EEG channel {ch+1}")
        nepochs = eeg_data.shape[0]
        signal = eeg_data[:, ch, :].flatten()
        # Apply EEG-specific preprocessing
        filtered_signal = highpass_filter(signal, config.HIGHPASS_FILTER_FREQ,eeg_fs)
        filtered_signal = notch_filter(filtered_signal, config.NOTCH_FILTER_FREQ, config.NOTCH_FILTER_Q, eeg_fs)
        filtered_signal = bandpass_filter(filtered_signal, config.BANDPASS_FILTER_LOWER_FREQ, config.BANDPASS_FILTER_HIGHER_FREQ, eeg_fs, config.BANDPASS_FILTER_ORDER)
        
        # FFT visualization
        y_range_signal = [-0.0002, 0.0002]
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        visualize_signal(signal[0:3750], eeg_fs, ax=axes[0,0], title=f"EEG Channel {ch+1} - Original Signal")
        axes[0,0].set_ylim(y_range_signal)
        visualize_fft(signal, eeg_fs, ax=axes[1,0], title=f"EEG Channel {ch+1} - Original Signal FFT")
        visualize_signal(filtered_signal[0:3750], eeg_fs, ax=axes[0,1], title=f"EEG Channel {ch+1} - Filtered Signal")
        axes[0,1].set_ylim(y_range_signal)
        visualize_fft(filtered_signal, eeg_fs, ax=axes[1,1], title=f"EEG Channel {ch+1} - Filtered Signal FFT")
        plt.tight_layout()
        fig.savefig(os.path.join(config.FIGURES_PREPROCESSING_DIR, f"EEG_filtering_channel_{ch+1}.png"))
        preprocessed_eeg[:, ch, :] = filtered_signal.reshape(nepochs, -1)

    return preprocessed_eeg
