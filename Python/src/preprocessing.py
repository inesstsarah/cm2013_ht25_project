from scipy.signal import butter, iirnotch, filtfilt
from sklearn.linear_model import LinearRegression
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

def lowpass_filter(signal, cutoff, fs, order=5):
    """
    Butterworth lowpass filter to remove high-frequency noise.

    Args:
        signal (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        filtered_signal (np.ndarray): The filtered signal.

    Example:
        >>> filtered_signal = lowpass_filter(signal, cutoff, fs, order)
    """

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
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


    if config.CURRENT_ITERATION > 2:  # EOG starts in iteration 2
        # Process EOG channels (2 channels) - may need different filtering
        eog_data = multi_channel_data['eog']
        preprocessed_data['eog'] = preprocess_eog_channel(eog_data,channel_info, config)

        # Process EMG channel (1 channel) - may need higher frequency preservation
        emg_data = multi_channel_data['emg']

        preprocessed_data['emg'] = preprocess_emg_channel(emg_data,channel_info, config)

        # EOG Artifact Removal
        preprocessed_data['eeg'] = remove_eog_artifacts(preprocessed_data['eeg'], preprocessed_data['eog'])

        #EMG Artifact Removal
        preprocessed_data['eeg'] = remove_emg_artifacts(preprocessed_data['eeg'], preprocessed_data['emg'], channel_info['eeg_fs'])
        print("Multi-channel preprocessing applied to EEG + EOG + EMG")
    else:
        print("Iteration 1-2: Processing EEG channels only")

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
        preprocessed_data = preprocess_eeg_channel(data, channel_info, config)


    elif config.CURRENT_ITERATION >= 3:
        print("TODO: Students should use multi-channel data format for iteration 3+")
        preprocessed_data = data  # Placeholder

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return preprocessed_data


def preprocess_eeg_channel(eeg_data,channel_info, config):
    """
    Preprocess EEG channel data.
    Args:
        eeg_data (np.ndarray): A 2D array of shape (n_epochs, n_samples, n_channels).
        channel_info (dict): Information about channel sampling frequencies.
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
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        visualize_signal(signal[0:3750], eeg_fs, ax=axes[0,0], title=f"EEG Channel {ch+1} - Original Signal")
        visualize_fft(signal, eeg_fs, ax=axes[1,0], title=f"EEG Channel {ch+1} - Original Signal FFT")
        visualize_signal(filtered_signal[0:3750], eeg_fs, ax=axes[0,1], title=f"EEG Channel {ch+1} - Filtered Signal")
        visualize_fft(filtered_signal, eeg_fs, ax=axes[1,1], title=f"EEG Channel {ch+1} - Filtered Signal FFT")
        plt.tight_layout()
        fig.savefig(os.path.join(config.FIGURES_PREPROCESSING_DIR, f"EEG_filtering_channel_{ch+1}.png"))
        preprocessed_eeg[:, ch, :] = filtered_signal.reshape(nepochs, -1)

    return preprocessed_eeg

def preprocess_eog_channel(eog_data,channel_info, config):
    """
    Preprocess EOG channel data.
    Args:
        eog_data (np.ndarray): A 2D array of shape (n_epochs, n_samples, n_channels).
        channel_info (dict): Information about channel sampling frequencies.
        config (module): The configuration module.

    Returns:
        np.ndarray: Preprocessed EOG data of same shape.

    Example:
        >>> preprocessed_eog = preprocess_eog_channel(eog_data, channel_info, config)
    """

    # Process EOG channels (2 channels)
    eog_fs = channel_info['eog_fs'] 
    print(f"EOG sampling frequency: {eog_fs} Hz")
    preprocessed_eog = np.zeros_like(eog_data)

    for ch in range(eog_data.shape[1]):
        print(f"Processing EOG channel {ch+1}")
        nepochs = eog_data.shape[0]
        signal = eog_data[:, ch, :].flatten()  # Convert to microvolts
        # Apply EOG-specific preprocessing
        filtered_signal = bandpass_filter(signal, config.EOG_BANDPASS_FILTER_LOWER_FREQ, config.EOG_BANDPASS_FILTER_HIGHER_FREQ, eog_fs, config.EOG_BANDPASS_FILTER_ORDER)
        
        # FFT visualization
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        visualize_signal(signal[0:3750], eog_fs, ax=axes[0,0], title=f"EOG Channel {ch+1} - Original Signal")
        visualize_fft(signal, eog_fs, ax=axes[1,0], title=f"EOG Channel {ch+1} - Original Signal FFT")
        visualize_signal(filtered_signal[0:3750], eog_fs, ax=axes[0,1], title=f"EOG Channel {ch+1} - Filtered Signal")
        visualize_fft(filtered_signal, eog_fs, ax=axes[1,1], title=f"EOG Channel {ch+1} - Filtered Signal FFT")
        plt.tight_layout()
        fig.savefig(os.path.join(config.FIGURES_PREPROCESSING_DIR, f"EOG_filtering_channel_{ch+1}.png"))
        preprocessed_eog[:, ch, :] = filtered_signal.reshape(nepochs, -1)

    return preprocessed_eog

def preprocess_emg_channel(emg_data,channel_info, config):
    """
    Preprocess EMG channel data.
    Args:
        emg_data (np.ndarray): A 2D array of shape (n_epochs, n_samples, n_channels).
        channel_info (dict): Information about channel sampling frequencies.
        config (module): The configuration module.

    Returns:
        np.ndarray: Preprocessed EMG data of same shape.

    Example:
        >>> preprocessed_emg = preprocess_emg_channel(emg_data, channel_info, config)
    """

    # Process EMG channels (1 channel)
    emg_fs = channel_info['emg_fs'] 
    print(f"EMG sampling frequency: {emg_fs} Hz")
    preprocessed_emg = np.zeros_like(emg_data)

    print(f"Processing EMG")
    nepochs = emg_data.shape[0]
    signal = emg_data[:, 0, :].flatten()  # Convert to microvolts
    # Apply EMG-specific preprocessing
    filtered_signal = highpass_filter(signal, config.EMG_HIGHPASS_FILTER_FREQ, emg_fs)
    filtered_signal = lowpass_filter(filtered_signal, config.EMG_LOWPASS_FILTER_FREQ, emg_fs)
   
    # FFT visualization
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    visualize_signal(signal[0:3750], emg_fs, ax=axes[0,0], title=f"EMG - Original Signal")
    visualize_fft(signal, emg_fs, ax=axes[1,0], title=f"EMG - Original Signal FFT")
    visualize_signal(filtered_signal[0:3750], emg_fs, ax=axes[0,1], title=f"EMG - Filtered Signal")
    visualize_fft(filtered_signal, emg_fs, ax=axes[1,1], title=f"EMG - Filtered Signal FFT")
    plt.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_PREPROCESSING_DIR, f"EMG_filtering_channel.png"))
    preprocessed_emg[:, 0, :] = filtered_signal.reshape(nepochs, -1)

    return preprocessed_emg

def remove_eog_artifacts(eeg_data, eog_data):
    """
    Removes EOG artifacts from EEG signals using Linear Regression.
    Model: EEG_clean = EEG_raw - (beta * EOG)

    Args:
        eeg_data (np.ndarray): Shape (n_epochs, n_channels, n_samples)
        eog_data (np.ndarray): Shape (n_epochs, n_channels, n_samples)

    Returns:
        np.ndarray: Cleaned EEG data.
        
    Example:
        >>> cleaned_eeg = remove_eog_artifacts(eeg_data, eog_data)
    """
    n_epochs, n_eeg_ch, _= eeg_data.shape
    
    cleaned_eeg = np.zeros_like(eeg_data)
    model = LinearRegression()

    for i in range(n_epochs):
        # Transpose EOG to shape (n_samples, n_eog_channels) for sklearn
        X_eog = eog_data[i].T 
        
        for ch in range(n_eeg_ch):
            y_eeg = eeg_data[i, ch, :]
            
            # Fit model: EEG ~ beta * EOG
            model.fit(X_eog, y_eeg)
            
            # Predict the artifact component
            artifact = model.predict(X_eog)
            
            # Subtract artifact
            cleaned_eeg[i, ch, :] = y_eeg - artifact
            
    return cleaned_eeg

def remove_emg_artifacts(eeg_data, emg_data, fs):
    """
    Adaptive filtering based on EMG power.
    If EMG power is high, apply a stricter low-pass filter to EEG.

    Args:
        eeg_data (np.ndarray): (n_epochs, n_channels, n_samples)
        emg_data (np.ndarray): (n_epochs, 1, n_samples)
        fs (int): EEG sampling frequency

    Returns:
        np.ndarray: cleaned EEG data

    Example:
        >>> cleaned_eeg = remove_emg_artifacts(eeg_data, emg_data,
    """
    cleaned_eeg = np.copy(eeg_data)
    n_epochs = eeg_data.shape[0]
    
    # Calculate EMG power metric per epoch
    emg_rms = np.sqrt(np.mean(emg_data[:, 0, :]**2, axis=1))
    
    # Determine Threshold (Mean + 1 STD of the dataset's EMG activity)
    threshold = np.mean(emg_rms) + np.std(emg_rms)
    
    print(f"EMG Adaptive Filter: Threshold set at {threshold*1e6:.4f} uV")
    count_affected = 0

    for i in range(n_epochs):
        if emg_rms[i] > threshold:
            count_affected += 1
            # Apply strict Low Pass to remove muscle noise
            for ch in range(eeg_data.shape[1]):
                cleaned_eeg[i, ch, :] = lowpass_filter(cleaned_eeg[i, ch, :], cutoff=20, fs=fs)
    
    print(f"EMG Artifacts: Stricter filtering applied to {count_affected}/{n_epochs} epochs.")
    
    return cleaned_eeg
