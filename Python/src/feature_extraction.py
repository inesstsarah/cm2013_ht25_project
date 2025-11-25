from typing import Any
import numpy as np
import scipy.stats   
from scipy.signal import welch
import nolds
from joblib import Parallel, delayed
from spectrum import arburg
import pywt
from math import log, e


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


# ---- WAVELET FEATURE EXTRACTION ----
def extract_sample_entropy(labels, base=None):
    """Function to compute the sample entropy of a signal
    Args:
        labels (np.ndarray): 1D array of signal values
        base (int, optional): Base of logarithm. Defaults to e.

    Returns:
        float: Sample entropy of the signal

    Example:
        >>> entropy = extract_sample_entropy(signal)
    """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0

    _,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent

def wavelet_decomposition(signal, wavelet_name):
    """Function to do wavelet decomposition on a signal
    
    Args:
        signal (np.ndarray): 1D array of a signal for wavelet decomposition
        wavelet_name (str): name of wavelet family and the number e.g 'coif1' or 'db1'

    Returns:
        coeff_arr (np.ndarray): Array of coefficients of different levels
    """
    # Get wavelet 
    wavelet = pywt.Wavelet(wavelet_name)
    # Do wavelet decomposition
    coeff_arr = pywt.wavedec(signal, wavelet)
    
    return(coeff_arr)


def wavelet_feature_extraction(coeff_arr):
    """Function to extract signals from wavelet coefficients from 
    the decomposition
    
    Args:
        coeff_arr (np.ndarray): 2D array of all coefficients from array
    
    Returns:
        wavelet_features (dict): A dictionary of all wavelet features"""

    wavelet_features = {}
    # Try extracting the entropy, and other statistics from every coefficient if possible
    #for coeff in coeff_arr:
        # Do the feature extraction here for every coeff
    coeff = coeff_arr[-1]
    energy = np.sum(coeff**2)
    wavelet_features["energy"] = energy

    # Get statistical moments
    # Mean
    mean = np.mean(coeff)
    wavelet_features["mean"] = mean

    # Standard Deviation
    stdev = np.std(coeff)
    wavelet_features["stdev"] = stdev

    # Skewness
    skewness = scipy.stats.skew(coeff) 
    wavelet_features["skewness"] = skewness

<<<<<<< HEAD
    # Kurtosis
    kurt = scipy.stats.kurtosis(coeff)
    wavelet_features["kurtosis"] = kurt

    # Entropy
    entropy = extract_sample_entropy(coeff)
    wavelet_features["entropy"] = entropy
=======
        # Entropy
        entropy = extract_sample_entropy(coeff_arr[i])
        wavelet_features[f"entropy_{i}"] = entropy
>>>>>>> main
    
    return(wavelet_features)


def wavelet_processing(epoch, wavelet_name):
    coeff_arr = wavelet_decomposition(signal = epoch, wavelet_name = wavelet_name)
    # print(f"Array of coeffs: {coeff_arr}")
    # print(f"Shape of array coeffs: {coeff_arr.shape}")
    wavelet_features = wavelet_feature_extraction(coeff_arr=coeff_arr)
    return(wavelet_features)


# ---- TIME DOMAIN FEATURE EXTRACTION ----
def extract_time_domain_features(epoch):
    """
    Extract time-domain features from a single epoch of signal data.

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

    # Signal energy and power:
    features['total_energy'] = np.sum(epoch**2)
    features['mean_power'] = np.mean(epoch**2)

    # Hjorth Parameters:
    features['hjorth_activity'] = extract_hjorth_activity(epoch)
    features['hjorth_mobility'] = extract_hjorth_mobility(epoch)
    features['hjorth_complexity'] = extract_hjorth_complexity(epoch)

    # Frequency-related Features:
    features['zero_crossings'] = np.sum(np.diff(np.sign(epoch)) != 0)
    
    # Complexity Feature:
    features['entropy'] = extract_sample_entropy(epoch)

    return features


#  ---- AR features computation ----
def _ar_compute_psd(epoch, fs, order, n_freqs):
    """
    Compute AR-based PSD using Burg's method.

    Returns frequencies (Hz) and one-sided PSD.
    """
    # Estimate AR coefficients and driving noise variance
    ar_coeffs, noise_var = arburg(epoch, order)[:2]

    # Frequency grid (0 .. fs/2)
    freqs = np.linspace(0.0, fs / 2.0, n_freqs)

    # Evaluate A(e^{-j 2π f k / fs}) over grid
    # ar_coeffs is [1, a1, ..., ap] as returned by arburg
    k_indices = np.arange(1, len(ar_coeffs))
    if len(k_indices) == 0:
        # Degenerate case
        psd = np.full_like(freqs, fill_value=noise_var / (1e-12))
        return freqs, psd

    exp_matrix = np.exp(-1j * 2.0 * np.pi * np.outer(freqs / fs, k_indices))  # shape (n_freqs, p)
    A_vals = 1.0 + exp_matrix @ ar_coeffs[1:]
    psd = noise_var / (np.abs(A_vals) ** 2)
    psd = np.real(psd)

    return freqs, psd


def _integrate_band_power(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> float:
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    return np.trapezoid(psd[mask], freqs[mask])


def _spectral_entropy(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> float:
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    p = psd[mask]
    p = np.clip(p, 1e-20, None)
    p = p / np.sum(p)
    H = -np.sum(p * np.log(p))
    H_norm = H / np.log(len(p))
    return float(H_norm)


def _spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float, percentile: float = 0.9) -> float:
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return float('nan')
    f_sel = freqs[mask]
    p_sel = psd[mask]
    cum = np.cumsum(p_sel)
    total = cum[-1]
    if total <= 0:
        return float('nan')
    target = percentile * total
    idx = np.searchsorted(cum, target)
    idx = np.clip(idx, 0, len(f_sel) - 1)
    return float(f_sel[idx])


def extract_ar_features(epoch: np.ndarray,
                        fs: int,
                        bands: dict,
                        order: int,
                        se_percentile: float,
                        config: Any) -> dict:
    """
    Extract AR (Burg) spectral features for a single EEG epoch.
<<<<<<< HEAD
=======

    Features:
    - Band powers (Delta/Theta/Alpha/Sigma/Beta)
    - Relative band powers (normalized by total power 0.5–30 Hz)
    - Spectral edge frequency (90% by default) within 0.5–30 Hz
    - Peak frequency within 0.5–30 Hz
    - Spectral entropy within 0.5–30 Hz
    """
    fmin_total = bands['delta'][0]
    fmax_total = bands['beta'][1]
    
    # Compute PSD and AR coefficients using pburg
    p = pburg(epoch, order=order, sampling=fs, criteria='AIC', NFFT=4096)
    psd = np.array(p.psd)
    freqs = np.array(p.frequencies())
>>>>>>> main

    Features:
    - Band powers (Delta/Theta/Alpha/Sigma/Beta)
    - Relative band powers (normalized by total power 0.5–30 Hz)
    - Spectral edge frequency (90% by default) within 0.5–30 Hz
    - Peak frequency within 0.5–30 Hz
    - Spectral entropy within 0.5–30 Hz
    """
    fmin_total, fmax_total = config.EEG_LOWER, config.EEG_UPPER
    
    freqs, psd = _ar_compute_psd(epoch, fs, order=order, n_freqs=1024)
    total_power = _integrate_band_power(freqs, psd, fmin_total, fmax_total)
    features = {}

    # Absolute and relative band powers
    for name, (f1, f2) in bands.items():
        bp = _integrate_band_power(freqs, psd, f1, f2)
        features[f'ar_{name}_power'] = bp
        features[f'ar_{name}_rel_power'] = (bp / total_power) if total_power > 0 else 0.0

    # Edge frequency, peak frequency, spectral entropy (within analysis band)
    features['ar_spectral_edge_freq'] = _spectral_edge_frequency(freqs, psd, fmin_total, fmax_total, se_percentile)

    mask = (freqs >= fmin_total) & (freqs <= fmax_total)
    if np.any(mask):
        peak_idx = np.argmax(psd[mask])
        features['ar_peak_frequency'] = float(freqs[mask][peak_idx])
    else:
        features['ar_peak_frequency'] = float('nan')

    features['ar_spectral_entropy'] = _spectral_entropy(freqs, psd, fmin_total, fmax_total)

    return features

# ---- WELCH METHOD ----
def welch_method(signal,fs, config):
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
    fs,    # Sampling frequency
    **config.WELCH_PARAMETERS            
    )

    return freqs, psd


def extract_welch_features(signal, fs, config):
    """ Function to extract spectral features from the PSD
    Args:   
        freqs (np.ndarray): 1D array of frequencies 
        psd (np.ndarray): 1D array of power spectral density values
        config (module): The configuration module.      
    Returns:
        spectral_features (dict): A dictionary of all spectral features
    Example:
        >>> spectral_features = extract_spectral_features(freqs, psd, config)
    """
    freqs, psd = welch_method(signal,fs, config)

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
        spectral_features['welch_'+band+'_power'] = band_power
       
        try:
            spectral_features['welch_'+band+'_power_rel'] = spectral_features['welch_'+band+'_power'] / total_power
        except ZeroDivisionError:
            spectral_features['welch_'+band+'_power_rel'] = 0.0
   

    #Spectral Entropy
    psd_norm = psd / np.sum(psd)
    spectral_entropy = (-np.sum(psd_norm * np.log2(psd_norm))) / np.log2(len(psd_norm))
    spectral_features['welch_spectral_entropy'] = spectral_entropy

    #Peak Frequency
    spectral_features['welch_peak_freq'] = freqs[np.argmax(psd)]

    #Spectral Edge Frequencies
    P90 = 0.9*total_power
    P95 = 0.95*total_power
    power_per_bin = psd*np.diff(freqs)[0]
    cumulative_power = np.cumsum(power_per_bin)
    spectral_features['welch_sef90'] = np.interp(P90, cumulative_power, freqs)
    spectral_features['welch_sef95']= np.interp(P95, cumulative_power, freqs)

    return spectral_features
<<<<<<< HEAD
=======

>>>>>>> main

def extract_features(data, channel_info, config):
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
        return extract_multi_channel_features(data,channel_info, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return extract_single_channel_features(data, channel_info, config)


def extract_multi_channel_features(multi_channel_data, channel_info, config):
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
            delayed(process_epoch)(i,multi_channel_data,channel_info,config) for i in range(n_epochs))
    else:
        for epoch_idx in range(n_epochs):
            print(f"Extracting EEG features for epoch {epoch_idx+1}/{n_epochs}")
            epoch_features = process_epoch(epoch_idx, multi_channel_data, channel_info, config)
            all_features.append(epoch_features)

    features = np.array(all_features)

    if config.CURRENT_ITERATION == 1:
        expected = 2 * 17  # 2 EEG channels × 17 features each
        print(f"Multi-channel Iteration 1: {features.shape[1]} features (target: {expected}+)")
    elif config.CURRENT_ITERATION >= 3:
        print(f"Multi-channel features extracted: {features.shape[1]} total")
        print("(2 EEG + 2 EOG + 1 EMG channels)")

    return features


def extract_single_channel_features(data, channel_info, config):
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
        expected = 2 * 17
        print(f"2 EEG channels Iteration 1: {features.shape[1]} features (target: {expected}+)")

    elif config.CURRENT_ITERATION == 2:
        # Iteration 2: Time + Frequency (AR) domain features
        print("Iteration 2: Adding AR spectral features")
        all_features = []
        for epoch in data:
            td = extract_time_domain_features(epoch)
            try:
                ar = extract_ar_features(epoch, channel_info['eeg_fs'], config.EEG_BANDS, config.AR_ORDER)
            except ImportError as e:
                print(str(e))
                ar = {}
            # Combine
            feat_vec = list(td.values()) + list(ar.values())
            all_features.append(feat_vec)
        features = np.array(all_features)

    elif config.CURRENT_ITERATION >= 3:
        # TODO: Students must implement multi-signal features
        print("TODO: Students should use multi-channel data format for iteration 3+")
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))  # Empty features - students must implement

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return features


def extract_eog_features(eog_signal): # TODO: make sure this function takes a 2D array of both EOG channels
    # TODO: Also
    """
    STUDENT TODO: Extract EOG-specific features for eye movement detection.

    EOG signals are used to detect:
    - Rapid eye movements (REM sleep indicator)
    - Slow eye movements
    - Eye blinks and artifacts
    """
    """
    Function to extract features from EOG signal
    Args:
        eog_signal (np.ndarray): 2D array of EOG Signal [EOG_left, EOG_right]
    Returns:
        eog_features (dict): Dictionary of EOG features
    """
    # Do highpass filter

    features = {
        'eog_mean_right': np.mean(eog_signal),
        'eog_std': np.std(eog_signal),
        'eog_range': np.max(eog_signal) - np.min(eog_signal),
        'eog_max': np.max(eog_signal),
        'eog_REM' : 0 # Change this

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


def process_epoch(epoch_idx, multi_channel_data, channel_info, config):
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
        eeg_signal = multi_channel_data['eeg'][epoch_idx, ch, :].copy()
        # Time-domain features
        eeg_td = extract_time_domain_features(eeg_signal)
        epoch_features.extend(list(eeg_td.values()))

        # Add AR spectral features
<<<<<<< HEAD
        eeg_ar = extract_ar_features(eeg_signal, channel_info['eeg_fs'], config.EEG_BANDS, config.AR_ORDER, config.EEG_SE_PERCENTILE, config)
=======
        eeg_ar = extract_ar_features(eeg_signal, channel_info['eeg_fs'], config.EEG_BANDS, config.AR_ORDER)
>>>>>>> main
        epoch_features.extend(list[Any](eeg_ar.values()))

        # Add wavelet features
        eeg_wavelet = wavelet_processing(eeg_signal, config.WAVELET_NAME)
        epoch_features.extend(list(eeg_wavelet.values()))

        # Add welch features
        eeg_welch = extract_welch_features(eeg_signal, channel_info['eeg_fs'], config)
        epoch_features.extend(list(eeg_welch.values()))

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




