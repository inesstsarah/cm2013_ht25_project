import numpy as np
import scipy.stats   
from scipy.signal import welch
import nolds
from joblib import Parallel, delayed

def extract_hjorth_activity(epoch):
    """
    Computes the Hjorth Activity for one epoch of signal data.

    Args:
        signal (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        float: A float representing the variance of the signal, also known as Hjorth Activity.
    Example:
        >>> activity = extract_hjorth_activity(epoch)
    """

    return np.var(epoch)

def extract_hjorth_mobility(epoch):
    """
    Computes the Hjorth Mobility for one epoch of signal data.

    Args:
        signal (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        float: A flaot representing the Hjorth Mobility.
    Example:
        >>> mobility = extract_hjorth_mobility(epoch)
    """

    return np.sqrt(extract_hjorth_activity(np.diff(epoch)) / extract_hjorth_activity(epoch))

def extract_hjorth_complexity(epoch):
    """
    Computes the Hjorth Complexity for one epoch of signal data.

    Args:
        signal (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        float: A Float representing the Hjorth Complexity.
    Example:
        >>> complexity = extract_hjorth_complexity(epoch)
    """

    return extract_hjorth_mobility(np.diff(epoch)) / extract_hjorth_mobility(epoch)

def extract_sample_entropy(epoch, m=2, r_factor=0.2):
    """
    Computes the Sample Entropy for one epoch of signal data.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.
        m (int): Embedding dimension.
        r_factor (float): Tolerance factor as a fraction of the standard deviation.

    Returns:
        entropy (float): Sample Entropy value.

    Example:
        >>> sampen = extract_sample_entropy(epoch)
    """
   
    r = r_factor * np.std(epoch)
    entropy = nolds.sampen(epoch, m, r)

    return entropy

def extract_time_domain_features(epoch):
    """
    Extract 17 time-domain features from a single epoch.

    Works for any signal type (EEG, EOG, EMG) but students should consider
    signal-specific features for optimal performance.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
       features (dict): A dictionary containing the extracted features.

    Example:
        >>> features = extract_time_domain_features(epoch)
    """

    features = {}

    # Statistical Moments:
    features['mean'] = np.mean(epoch)
    features['median'] = np.median(epoch)
    features['std'] = np.std(epoch)
    features['variance'] = np.var(epoch)
    features['skewness'] = scipy.stats.skew(epoch)
    features['kurtosis'] = scipy.stats.kurtosis(epoch)

    # Amplitude Features:
    features['rms'] = np.sqrt(np.mean(epoch**2))
    features['min'] = np.min(epoch)
    features['max'] = np.max(epoch)
    features['range'] = np.max(epoch) - np.min(epoch)
    features['total_energy'] = np.sum(epoch**2)

    # Hjorth Parameters:
    features['hjorth_activity'] = extract_hjorth_activity(epoch)
    features['hjorth_mobility'] = extract_hjorth_mobility(epoch)
    features['hjorth_complexity'] = extract_hjorth_complexity(epoch)

    # Frequency-related Features:
    features['zero_crossings'] = np.sum(np.diff(np.sign(epoch)) != 0)
    
    # Complexity Feature:
    # features['entropy'] = extract_sample_entropy(epoch)

    return features

def extract_features(data, config):
    """
    Extract features from the preprocessed data.

    This function should handle both single-channel (old format) and
    multi-channel data (new format with 2 EEG + 2 EOG + 1 EMG channels).

    Iteration 1: 16 time-domain features per EEG channel
    Iteration 2: 31+ features (time + frequency domain) per channel
    Iteration 3: Multi-signal features (EEG + EOG + EMG)
    Iteration 4: Optimized feature set (selected subset)

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        config (module): The configuration module.

    Returns:
        np.ndarray: A 2D array of features (n_epochs, n_features).

    Example:
        >>> features = extract_features(preprocessed_data, config)
    """

    print(f"Extracting features for iteration {config.CURRENT_ITERATION}...")

    # Detect if we have multi-channel data structure
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return extract_multi_channel_features(data, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return extract_single_channel_features(data, config)


def extract_multi_channel_features(multi_channel_data, config):
    """
    Extract features from multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.

    Args:
        multi_channel_data (dict): Dictionary with keys 'eeg', 'eog', 'emg'.
        config (module): The configuration module.

    Returns:
        features (np.ndarray): A 2D array of features (n_epochs, n_features).
    
    Example:
        >>> features = extract_multi_channel_features(multi_channel_data, config)
    """

    print("selecting multi-channel features...")
    
    n_epochs = multi_channel_data['eeg'].shape[0]
    all_features = []
    
    if config.USE_PARALLEL:
        all_features = Parallel(n_jobs=config.PARALLEL_N_JOBS, backend='loky', verbose=10)(
            delayed(process_epoch)(i,multi_channel_data,config) for i in range(n_epochs))
    else:
        for epoch_idx in range(n_epochs):
            #print(f"Extracting EEG features for epoch {epoch_idx+1}/{n_epochs}")
            epoch_features = process_epoch(epoch_idx,multi_channel_data,config)
            all_features.append(epoch_features)

    features = np.array(all_features)

    if config.CURRENT_ITERATION == 1:
        expected = 2 * 16  # 2 EEG channels × 16 features each
        print(f"Multi-channel Iteration 1: {features.shape[1]} features (target: {expected}+)")  
    elif config.CURRENT_ITERATION >= 3:
        print(f"Multi-channel features extracted: {features.shape[1]} total")
        print("(2 EEG + 2 EOG + 1 EMG channels)")

    return features


def extract_single_channel_features(data, config):
    """
    Extract features from single-channel data for backward compatibility.

    Args:

        data (np.ndarray): A 2D array of shape (n_epochs, n_samples).
        config (module): The configuration module.  

    Returns:
        features (np.ndarray): A 2D array of features (n_epochs, n_features).

    Example:
        >>> features = extract_single_channel_features(single_channel_data, config)
    """
    if config.CURRENT_ITERATION == 1:
        # Iteration 1: Time-domain features (TARGET: 16 features)
        all_features = []
        for epoch in data:
            features = extract_time_domain_features(epoch)
            all_features.append(list(features.values()))
        features = np.array(all_features)
        expected = 2 * 16
        print(f"2 EEG channels Iteration 1: {features.shape[1]} features (target: {expected}+)")

    elif config.CURRENT_ITERATION == 2:
        # TODO: Students must implement frequency-domain features
        print("TODO: Students must implement frequency-domain feature extraction")
        print("Target: ~31 features (time + frequency domain)")
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))  # Empty features - students must implement

    elif config.CURRENT_ITERATION >= 3:
        # TODO: Students must implement multi-signal features
        print("TODO: Students should use multi-channel data format for iteration 3+")
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))  # Empty features - students must implement

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return features

def extract_eog_features(eog_signal):
    """
    STUDENT TODO: Extract EOG-specific features for eye movement detection.

    EOG signals are used to detect:
    - Rapid eye movements (REM sleep indicator)
    - Slow eye movements
    - Eye blinks and artifacts
    """
    features = {
        'eog_mean': np.mean(eog_signal),
        'eog_std': np.std(eog_signal),
        'eog_range': np.max(eog_signal) - np.min(eog_signal),
    }

    # TODO: Students should add:
    # - Eye movement detection features
    # - Rapid vs slow movement discrimination
    # - Cross-channel correlations (left vs right eye)

    return features


def extract_emg_features(emg_signal):
    """
    STUDENT TODO: Extract EMG-specific features for muscle tone detection.

    EMG signals are used to detect:
    - Muscle tone levels (high in wake, low in REM)
    - Muscle twitches and artifacts
    - Sleep-related muscle activity
    """
    features = {
        'emg_mean': np.mean(emg_signal),
        'emg_std': np.std(emg_signal),
        'emg_rms': np.sqrt(np.mean(emg_signal**2)),
    }

    # TODO: Students should add:
    # - High-frequency power (muscle activity indicator)
    # - Spectral edge frequency
    # - Muscle tone quantification

    return features

def process_epoch(epoch_idx, multi_channel_data, config):
    """
    Process a single epoch to extract features from EEG, EOG, and EMG channels.

    Args:
        epoch_idx (int): The index of the epoch to process.
        multi_channel_data (dict): Dictionary with keys 'eeg', 'eog', 'emg'.
        config (module): The configuration module.
    Returns:
        epoch_features (list): A list of extracted features for the epoch.
    
    Example:
        >>> epoch_features = process_epoch(epoch_idx, multi_channel_data, config)
    """
    epoch_features = []
    # EEG features (2 channels)
    for ch in range(multi_channel_data['eeg'].shape[1]):
        eeg_signal = multi_channel_data['eeg'][epoch_idx, ch, :]
        eeg_features = extract_time_domain_features(eeg_signal)
        epoch_features.extend(list(eeg_features.values()))
        freqs, psd = welch_method(eeg_signal,config= config)
        eeg_features = extract_spectral_features(freqs, psd, config)
        epoch_features.extend(list(eeg_features.values()))

    if config.CURRENT_ITERATION >= 3:
        # Add EOG features (2 channels)
        for ch in range(multi_channel_data['eog'].shape[1]):
            eog_signal = multi_channel_data['eog'][epoch_idx, ch, :]
            eog_features = extract_eog_features(eog_signal)
            epoch_features.extend(list(eog_features.values()))

        # Add EMG features (1 channel)
        emg_signal = multi_channel_data['emg'][epoch_idx, 0, :]
        emg_features = extract_emg_features(emg_signal)
        epoch_features.extend(list(emg_features.values()))

    return epoch_features

def welch_method(signal, config):
    """
    Computes the Power Spectral Density (PSD) using Welch's method.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.
        fs (int): Sampling frequency of the signal.
        nperseg (int): Length of each segment for Welch's method.
        noverlap (int): Number of overlapping samples between segments.
        nfft (int): Number of points for the FFT.
        window (str or tuple or array_like): Desired window to use.
        scaling (str): Selects between 'density' and 'spectrum' scaling of the PSD.

    Returns:
        f (np.ndarray): Array of sample frequencies.
        Pxx (np.ndarray): Power spectral density of the signal.

    Example:
        >>> freqs, psd = welch_psd(epoch, fs=100, nperseg=256)
    """
   
    freqs, psd = welch(
    signal,
    config.EEG_FS,    # Sampling frequency
    **config.WELCH_PARAMETERS            
    )

    return freqs, psd

def extract_spectral_features(freqs, psd, config):
    spectral_features = {}
    indices = np.where((freqs>=config.EEG_LOWER)&(freqs<=config.EEG_UPPER))
    freqs = freqs[indices]
    psd = psd[indices]
    total_power = np.trapezoid(psd, freqs)

    #Band Powers
    for band,(lower,upper) in config.EEG_BANDS.items():
        band_i = np.where((freqs>=lower) & (freqs<=upper))
        band_freqs = freqs[band_i]
        band_psd = psd[band_i]
        band_power = np.trapezoid(band_psd,band_freqs)
        spectral_features[band+'_power'] = band_power
       
        try:
            spectral_features[band+'_power_rel'] = spectral_features[band+'_power'] / total_power
        except ZeroDivisionError:
            spectral_features[band+'_power_rel'] = 0.0
   

    #Spectral Entropy
    psd_norm = psd / np.sum(psd)
    spectral_entropy = (-np.sum(psd_norm * np.log2(psd_norm))) / np.log2(len(psd_norm))
    spectral_features['spectral_entropy'] = spectral_entropy

    #Peak Frequency
    spectral_features['peak_freq'] = freqs[np.argmax(psd)]

    #Spectral Edge Frequencies
    P90 = 0.9*total_power
    P95 = 0.95*total_power
    power_per_bin = psd*np.diff(freqs)[0]
    cumulative_power = np.cumsum(power_per_bin)
    spectral_features['sef90'] = np.interp(P90, cumulative_power, freqs)
    spectral_features['sef95']= np.interp(P95, cumulative_power, freqs)

    return spectral_features

    

        

    

    


    