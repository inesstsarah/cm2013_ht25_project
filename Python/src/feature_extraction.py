import numpy as np
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
def extract_hjorth_activity(epoch):
    """
    Computes the Hjorth Activity for one epoch of signal data.

    Args:
        signal (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        float: A float representing the variance of the signal, also known as Hjorth Activity.
    """
    return np.var(epoch)

def extract_hjorth_mobility(epoch):
    """
    Computes the Hjorth Mobility for one epoch of signal data.

    Args:
        signal (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        float: A flaot representing the Hjorth Mobility.
    """
    return np.sqrt(extract_hjorth_activity(np.diff(epoch)) / extract_hjorth_activity(epoch))

def extract_hjorth_complexity(epoch):
    """
    Computes the Hjorth Complexity for one epoch of signal data.

    Args:
        signal (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        float: A Float representing the Hjorth Complexity.
    """

    return extract_hjorth_mobility(np.diff(epoch)) / extract_hjorth_mobility(epoch)

def extract_simple_k_complex(epoch):
    """
    
    Simple K-complex detection i time domain.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        nb (int): Number of K-complexes detected
        float : Total duration of K-complexes in epoch in seconds
    """
    eeg_fs = 125
    min_delta = int(0.5*eeg_fs)
    max_delta = int(1.5*eeg_fs)
    min_peak_to_peak = 75

    #Find peaks that stands out with at least 30 uV and is 240 ms from next peak
    neg_peaks,_ = scipy.signal.find_peaks(-epoch,prominence = 30, distance = 30)
    pos_peaks,_ = scipy.signal.find_peaks(epoch,prominence = 30, distance = 30)
    #print(len(pos_peaks)," ", len(neg_peaks))
    pos_idx = 0
    k_complexes=[]
    for neg_idx in neg_peaks:
        while pos_idx < len(pos_peaks) and pos_peaks[pos_idx]<= neg_idx:
            pos_idx +=1
        for j in range(pos_idx,len(pos_peaks)):
            delta = pos_peaks[j]-neg_idx
            peak_to_peak = np.abs(epoch[pos_peaks[j]]-epoch[neg_idx])
            if delta > max_delta:
                break
            if delta >= min_delta  and peak_to_peak > min_peak_to_peak:
                k_complexes.append(
                    {
                        'delta' : delta,
                        'peak_to_peak' : peak_to_peak
                    }
                )
    nb = len(k_complexes)
    duration = 0
    for complex in k_complexes:
        duration += complex['delta']
    return nb, duration/eeg_fs

def extract_sample_entropy(epoch, m=2, r_factor=0.2):
    """
    Computes the Sample Entropy for one epoch of signal data.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.
        m (int): Embedding dimension.
        r_factor (float): Tolerance factor as a fraction of the standard deviation.

    Returns:
        float: Sample Entropy value.
    """
   
    r = r_factor * np.std(epoch)
    return nolds.sampen(epoch, m, r)

def extract_time_domain_features(epoch):
    """
    Extract 17 time-domain features from a single epoch.

    Works for any signal type (EEG, EOG, EMG) but students should consider
    signal-specific features for optimal performance.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        dict: A dictionary of features.
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
    features['mean_power'] = np.mean(epoch**2)

    # Hjorth Parameters:
    features['hjorth_activity'] = extract_hjorth_activity(epoch)
    features['hjorth_mobility'] = extract_hjorth_mobility(epoch)
    features['hjorth_complexity'] = extract_hjorth_complexity(epoch)

    # Frequency-related Features:
    features['zero_crossings'] = np.sum(np.diff(np.sign(epoch)) != 0)
    
    # Complexity Feature:
    #features['Entropy'] = extract_sample_entropy(epoch)

    #K-complex features:
    #features['nb_complexes'],  features['duration_complexes'] = simple_k_complex_extraction(epoch)

    return features

def extract_features(data, config):
    """
    STUDENT IMPLEMENTATION AREA: Extract features based on current iteration.

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
        np.ndarray: A 2D array of features (n_epochs, n_features).
    """
    print("selecting multi-channel features...")
    def process_epoch(epoch_idx):
        epoch_features = []
        # EEG features (2 channels)
        for ch in range(multi_channel_data['eeg'].shape[1]):
            eeg_signal = multi_channel_data['eeg'][epoch_idx, ch, :]
            eeg_features = extract_time_domain_features(eeg_signal)
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

    n_epochs = multi_channel_data['eeg'].shape[0]
    all_features = []
    
    if config.USE_PARALLEL:
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for parallel processing, but not installed.")
        
        all_features = Parallel(n_jobs=config.PARALLEL_N_JOBS, backend='loky', verbose=10)(
            delayed(process_epoch)(i) for i in range(n_epochs))
    else:
        for epoch_idx in range(n_epochs):
            print(f"Extracting EEG features for epoch {epoch_idx+1}/{n_epochs}")
            epoch_features = process_epoch(epoch_idx)
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
        np.ndarray: A 2D array of features (n_epochs, n_features).
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
