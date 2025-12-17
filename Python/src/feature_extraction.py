from typing import Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (must be set before pyplot import)
import numpy as np
import scipy.stats   
from scipy.signal import welch
from joblib import Parallel, delayed
from spectrum import arburg, pburg
import pywt
from math import log, e
import os
from src.preprocessing import highpass_filter, bandpass_filter
from src.utils import cross_correlation
from scipy.signal import find_peaks


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
    Example:
        >>> coeff_arr = wavelet_decomposition(signal, wavelet_name)
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
        wavelet_features (dict): A dictionary of all wavelet features
    
    Example:
        >>> wavelet_features = wavelet_feature_extraction(coeff_arr)
    """

    wavelet_features = {}
    # Try extracting the entropy, and other statistics from every coefficient if possible
    for i in range(len(coeff_arr)):
    #for coeff in coeff_arr:
        # Do the feature extraction here for every coeff
        energy = np.sum(coeff_arr[i]**2)
        wavelet_features[f"energy_{i}"] = energy
    
        # Get statistical moments
        # Mean
        mean = np.mean(coeff_arr[i])
        wavelet_features[f"mean_{i}"] = mean

        # Standard Deviation
        stdev = np.std(coeff_arr[i])
        wavelet_features[f"stdev_{i}"] = stdev

        # Skewness
        skewness = scipy.stats.skew(coeff_arr[i]) 
        wavelet_features[f"skewness_{i}"] = skewness

        # Kurtosis
        kurt = scipy.stats.kurtosis(coeff_arr[i])
        wavelet_features[f"kurtosis_{i}"] = kurt

        # Entropy
        entropy = extract_sample_entropy(coeff_arr[i])
        wavelet_features[f"entropy_{i}"] = entropy
    
    return(wavelet_features)

def wavelet_processing(epoch, wavelet_name):
    """Function to do wavelet processing on a signal epoch
    Args:
        epoch (np.ndarray): 1D array of a signal epoch
        wavelet_name (str): name of wavelet family and the number e.g 'coif1' or 'db1'
    Returns:
        wavelet_features (dict): A dictionary of all wavelet features
    Example:  
        >>> wavelet_features = wavelet_processing(epoch, wavelet_name)
    """
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

    # Hjorth Parameters:
    features['hjorth_activity'] = extract_hjorth_activity(epoch)
    features['hjorth_mobility'] = extract_hjorth_mobility(epoch)
    features['hjorth_complexity'] = extract_hjorth_complexity(epoch)

    # Frequency-related Features:
    features['zero_crossings'] = np.sum(np.diff(np.sign(epoch)) != 0)
    
    # Complexity Feature:
    features['entropy'] = extract_sample_entropy(epoch)

    return features


# ========== AR features computation ========== {
def _integrate_band_power(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> float:
    """ Compute the band power by integrating the PSD within a frequency band.
    Args:
        freqs (np.ndarray): Frequency array (Hz).
        psd (np.ndarray): Power spectral density array.
        f_low (float): Lower frequency bound (Hz).
        f_high (float): Upper frequency bound (Hz).
    Returns:
        float: Band power.
    Example:
        >>> band_power = _integrate_band_power(freqs, psd, f_low, f_high)
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    return np.trapezoid(psd[mask], freqs[mask])


def _peak_frequency(freqs: np.ndarray, psd: np.ndarray, f_low: float = None, f_high: float = None) -> float:
    """ Compute the peak frequency of the PSD within a frequency band.
    Args:
        freqs (np.ndarray): Frequency array (Hz).
        psd (np.ndarray): Power spectral density array.
        f_low (float, optional): Lower frequency bound (Hz). Defaults to None.
        f_high (float, optional): Upper frequency bound (Hz). Defaults to None.
    Returns:
        float: Peak frequency (Hz).
    Example:
        >>> peak_freq = _peak_frequency(freqs, psd, f_low, f_high)
    """
    if f_low is not None and f_high is not None:
        mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(mask):
            return float('nan')
        freqs_sel = freqs[mask]
        psd_sel = psd[mask]
    else:
        freqs_sel = freqs
        psd_sel = psd
    
    if len(psd_sel) == 0 or np.max(psd_sel) <= 0:
        return float('nan')
    
    peak_idx = np.argmax(psd_sel)
    return float(freqs_sel[peak_idx])


def _spectral_entropy(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> float:
    """ Compute the spectral entropy of the PSD within a frequency band.
    Args:
        freqs (np.ndarray): Frequency array (Hz).
        psd (np.ndarray): Power spectral density array.
        f_low (float): Lower frequency bound (Hz).
        f_high (float): Upper frequency bound (Hz).
    Returns:
        float: Spectral entropy (normalized).
    Example:
        >>> entropy = _spectral_entropy(freqs, psd, f_low, f_high)
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    p = psd[mask]
    p = np.clip(p, 1e-20, None)
    p = p / np.sum(p)
    H = -np.sum(p * np.log(p))
    H_norm = H / np.log(len(p))
    return float(H_norm)


def _extract_derivative_features(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> dict:
    """
    Extract derivative features from power spectral density.
    
    Calculates first and second order derivatives of PSD and extracts
    statistical features (mean, std, max, min, range, power).
    
    Args:
        freqs (np.ndarray): Frequency array (Hz).
        psd (np.ndarray): Power spectral density array.
    
    Returns:
        dict: Dictionary containing derivative features.
    Example:
        >>> deriv_features = _extract_derivative_features(freqs, psd, f_low, f_high)
    """
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return {}
    psd = psd[mask]
    freqs = freqs[mask]
    features = {}
    
    # Calculate frequency step
    df = np.diff(freqs)[0] if len(np.diff(freqs)) > 0 else (freqs[-1] - freqs[0]) / len(freqs)
    
    # First-order derivative (dPSD/df) - reflects rate of change of power with frequency
    psd_first_derivative = np.diff(psd) / df
    
    # First-order derivative statistics
    features['deriv1_mean'] = np.mean(psd_first_derivative)
    features['deriv1_std'] = np.std(psd_first_derivative)
    features['deriv1_max'] = np.max(psd_first_derivative)
    features['deriv1_min'] = np.min(psd_first_derivative)

    # First-order derivative power (integral of squared derivative) - measures total spectral variation
    features['deriv1_power'] = np.trapezoid(psd_first_derivative**2, freqs[1:])
    
    # Second-order derivative (d²PSD/df²) - reflects spectral curvature/sharpness
    psd_second_derivative = np.diff(psd_first_derivative) / df
    
    # Second-order derivative statistics
    features['deriv2_mean'] = np.mean(psd_second_derivative)
    features['deriv2_std'] = np.std(psd_second_derivative)
    features['deriv2_max'] = np.max(psd_second_derivative)
    features['deriv2_min'] = np.min(psd_second_derivative)
    
    # Second-order derivative power
    features['deriv2_power'] = np.trapezoid(psd_second_derivative**2, freqs[2:])
    
    return features


def _spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, f_low: float, f_high: float, percentile: float = 0.9) -> float:
    """ Compute the spectral edge frequency (SEF) at a given percentile within a frequency band.
    Args:
        freqs (np.ndarray): Frequency array (Hz).
        psd (np.ndarray): Power spectral density array.
        f_low (float): Lower frequency bound (Hz).
        f_high (float): Upper frequency bound (Hz).
        percentile (float, optional): Percentile for SEF calculation. Defaults to 0.9.
    Returns:
        float: Spectral edge frequency (Hz).
    Example:
        >>> sef = _spectral_edge_frequency(freqs, psd, f_low, f_high, percentile)
    """
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
                        order: int) -> dict:
    """
    Extract AR (Burg) spectral features for a single EEG epoch.

    Features:
    - Band powers (Delta/Theta/Alpha/Sigma/Beta)
    - Relative band powers (normalized by total power 0.5–30 Hz)
    - Spectral edge frequency (90% by default) within 0.5–30 Hz
    - Peak frequency within 0.5–30 Hz
    - Spectral entropy within 0.5–30 Hz

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of EEG signal data.
        fs (int): Sampling frequency of the signal.
        bands (dict): Dictionary defining frequency bands.
        order (int): Order of the AR model.

    Returns:
        dict: A dictionary containing the extracted AR spectral features.

    Example:
        >>> features = extract_ar_features(epoch, fs, bands, order)
    """
    fmin_total = bands['delta'][0]
    fmax_total = bands['beta'][1]
    
    # Compute PSD and AR coefficients using pburg
    p = pburg(epoch, order=order, sampling=fs, criteria='AIC', NFFT=4096)
    psd = np.array(p.psd)
    freqs = np.array(p.frequencies())

    total_power = _integrate_band_power(freqs, psd, fmin_total, fmax_total)
    features = {}

    # Absolute and relative band powers
    for name, (f1, f2) in bands.items():
        bp = _integrate_band_power(freqs, psd, f1, f2)
        features[f'ar_{name}_power'] = bp
        features[f'ar_{name}_rel_power'] = (bp / total_power) if total_power > 0 else 0.0
        # Peak frequency within each band
        features[f'ar_{name}_peak_freq'] = _peak_frequency(freqs, psd, f1, f2)

    # Spectral edge frequency (SEF90)
    features['ar_spectral_edge_freq'] = _spectral_edge_frequency(freqs, psd, fmin_total, fmax_total, 0.9)
    # Global peak frequency
    features['ar_peak_frequency'] = _peak_frequency(freqs, psd, fmin_total, fmax_total)
    # Spectral entropy
    features['ar_spectral_entropy'] = _spectral_entropy(freqs, psd, fmin_total, fmax_total)
    # Derivative features
    deriv_features = _extract_derivative_features(freqs, psd, fmin_total, fmax_total)
    features.update(deriv_features)

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
    # OPTIMIZATION: Pre-extract epoch data to avoid any slicing/copying in worker processes
    # Extract all channel data upfront to minimize work inside parallel workers
    epoch_data_list = [
        {
            'eeg': multi_channel_data['eeg'][epoch_idx, :, :],
            'eog': multi_channel_data['eog'][epoch_idx, :, :] if config.CURRENT_ITERATION >= 3 else None,
            'emg': multi_channel_data['emg'][epoch_idx, :, :] if config.CURRENT_ITERATION >= 3 else None
        }
        for epoch_idx in range(n_epochs)
    ]

    if config.USE_PARALLEL:
        # from threadpoolctl import threadpool_limits
        # threadpool_limits(limits=1)
        print(f"Preparing {n_epochs} epochs for parallel processing...")
        n_jobs = config.PARALLEL_N_JOBS if config.PARALLEL_N_JOBS > 0 else os.cpu_count() or 4
        batch_size = max(1, n_epochs // (n_jobs * 3))
        print(f"Using {n_jobs} workers with batch_size: {batch_size}")
        all_features = Parallel(
            n_jobs=config.PARALLEL_N_JOBS, 
            backend='loky', 
            verbose=10,
            prefer='processes',
            batch_size=batch_size
        )(
            delayed(_process_epoch)(epoch_data, channel_info, config) 
            for epoch_data in epoch_data_list
        )
    else:
        all_features = []
        for epoch_idx in range(n_epochs):
            print(f"Extracting features for epoch {epoch_idx+1}/{n_epochs}")
            epoch_features = _process_epoch(epoch_data_list[epoch_idx], channel_info, config)
            all_features.append(epoch_features)

    features = np.array(all_features)
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



def extract_eog_features(eog_signal, fs): # TODO: Make sure that there is a 2D array for the input
    """
    STUDENT TODO: Extract EOG-specific features for eye movement detection.

    EOG signals are used to detect:
    - Rapid eye movements (REM sleep indicator)
    - Slow eye movements
    - Eye blinks and artifacts
    """
    """
    Function to extract EOG features from the EOG signal

    Args:
        eog_signal (np.ndarray): 2D array with the EOG signal as follows: [left_eog, right_eog])
        fs (int): Sampling frequency of signal

    Returns:
        features (dict): feature dictionary of EOG signal
    
    Example:
        >>> features = extract_eog_features(eog_signal, fs)
        
    """
    
    eog_signal_L = eog_signal[0]
    eog_signal_R = eog_signal[1]

    # Get highpass filter for REM detection
    filtered_eog_R = bandpass_filter(eog_signal_R,0.5,5,fs,4)
    filtered_eog_L = bandpass_filter(eog_signal_L,0.5,5,fs,4)
    neg_product = -(filtered_eog_L * filtered_eog_R)
    peaks = find_peaks(neg_product, distance = 75, height = 0) # Spacing: 75 samples/0.5 seconds from each other
    peak = peaks[0]
    num_peaks = len(peak)


    cross_corr = cross_correlation(eog_signal_L, eog_signal_R)
    


    features = {
        'eog_mean_L': np.mean(eog_signal_L),
        'eog_std_L': np.std(eog_signal_L),
        'eog_range_L': np.max(eog_signal_L) - np.min(eog_signal_L),
        'eog_max_L': np.max(eog_signal_L),

        'eog_mean_R': np.mean(eog_signal_R),
        'eog_std_R': np.std(eog_signal_R),
        'eog_range_R': np.max(eog_signal_R) - np.min(eog_signal_R),
        'eog_max_R': np.max(eog_signal_R),
        
        'cross_corr': cross_corr,
        'REM_peaks': num_peaks
    }

    return features


def extract_emg_features(emg_signal: np.ndarray, fs: int, bands: dict)->dict:
    """
    Extract EMG-specific features for muscle tone detection.

    EMG signals are used to detect:
    - Muscle tone levels (high in wake, low in REM)
    - Muscle twitches and artifacts
    - Sleep-related muscle activity

    Args:
        emg_signal (np.ndarray): A 1D array representing one epoch of EMG
        fs (int): Sampling frequency of the signal.
        bands (dict): Dictionary defining frequency bands.
    Returns:
        dict: A dictionary containing the extracted EMG features.
    Example:
        >>> features = extract_emg_features(emg_signal, fs, bands)
    """
    fmin_total = bands['delta'][0]
    fmax_total = bands['beta'][1]

    features = {
        'emg_mean': np.mean(emg_signal),
        'emg_std': np.std(emg_signal),
        'emg_rms': np.sqrt(np.mean(emg_signal**2)),
        'emg_power': np.mean(emg_signal**2), # Signal power (mean squared amplitude)
        'emg_variance': np.var(emg_signal),  # Variance
    }

    # TODO: Students should add:
    # - High-frequency power (muscle activity indicator)
    # - Spectral edge frequency
    # - Muscle tone quantification

    # High-frequency (20-40 Hz) power ratio
    nperseg = min(len(emg_signal), fs * 2)
    freqs, psd = welch(emg_signal, fs=fs, nperseg=nperseg)
    
    # Total power
    total_power = np.sum(psd)
    
    # High-frequency power (20-40 Hz)
    hf_mask = (freqs >= 20) & (freqs <= 40)
    hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 0
    
    features['emg_hf_power'] = hf_power
    features['emg_hf_ratio'] = hf_power / (total_power + 1e-10)

    # Spectral edge frequency
    features['emg_spectral_edge_freq'] = _spectral_edge_frequency(freqs, psd, fmin_total, fmax_total, 0.9)

    return features


def _process_epoch(epoch_data: dict, channel_info: dict, config: dict) -> list:
    """
    Process a single epoch's data to extract features.
    This is the core processing function that extracts all features from epoch data.
    
    Args:
        epoch_data (dict): Dictionary with 'eeg', 'eog', 'emg' keys containing arrays.
                          - 'eeg': shape (n_channels, n_samples)
                          - 'eog': shape (n_channels, n_samples)
                          - 'emg': shape (n_samples,)
        channel_info (dict): Channel information (e.g., 'eeg_fs').
        config: Config object or proxy with necessary parameters.

    Returns:
        epoch_features (list): List of extracted features for the epoch.

    Example:
        >>> epoch_features = _process_epoch(epoch_data, channel_info, config)
    """
    epoch_features = []
    eeg_data = epoch_data['eeg']  # shape: (n_channels, n_samples)
    
    # EEG features (typically 2 channels)
    for ch in range(eeg_data.shape[0]):
        eeg_signal = eeg_data[ch, :]
        # Time-domain features
        eeg_td = extract_time_domain_features(eeg_signal)
        epoch_features.extend(list(eeg_td.values()))
    
        if config.CURRENT_ITERATION >= 2:
            # Add AR spectral features
            eeg_ar = extract_ar_features(eeg_signal, channel_info['eeg_fs'], config.EEG_BANDS, config.AR_ORDER)
            epoch_features.extend(list[Any](eeg_ar.values()))

            # Add wavelet features
            eeg_wavelet = wavelet_processing(eeg_signal, config.WAVELET_NAME)
            epoch_features.extend(list(eeg_wavelet.values()))

            # Add welch features
            eeg_welch = extract_welch_features(eeg_signal, channel_info['eeg_fs'], config)
            epoch_features.extend(list(eeg_welch.values()))

    if config.CURRENT_ITERATION >= 3:
        # Add EOG features (2 channels)
        eog_data = epoch_data['eog']
        eog_features = extract_eog_features(eog_signal=eog_data, fs = channel_info['eog_fs'])
        epoch_features.extend(list(eog_features.values()))

        # Add EMG features (1 channel)
        emg_signal = epoch_data['emg'][0, :]
        emg_features = extract_emg_features(emg_signal, channel_info['emg_fs'], config.EEG_BANDS)
        epoch_features.extend(list(emg_features.values()))

    return epoch_features


def get_feature_names(multi_channel_data: dict, channel_info: dict, config: dict) -> list:
    """
    Get the feature names for the current iteration.
    Args:
        multi_channel_data (dict): Dictionary with keys 'eeg', 'eog', 'emg'.
        channel_info (dict): Channel information (e.g., 'eeg_fs').
        config: Config object or proxy with necessary parameters.
    Returns:
        feature_names (list): List of feature names.
        
    Example:
        >>> feature_names = get_feature_names(multi_channel_data, channel_info, config)
    """
    feature_names = []
    if config.CURRENT_ITERATION == 1:
        for ch in range(multi_channel_data['eeg'].shape[1]):
            ch_feature_names = []
            ch_feature_names.extend(list(extract_time_domain_features(multi_channel_data['eeg'][0, ch, :]).keys()))
            ch_feature_names = [f"eeg_{ch}_{name}" for name in ch_feature_names]
            feature_names.extend(ch_feature_names)
    elif config.CURRENT_ITERATION == 2:
        for ch in range(multi_channel_data['eeg'].shape[1]):
            ch_feature_names = []
            ch_feature_names.extend(list(extract_time_domain_features(multi_channel_data['eeg'][0, ch, :]).keys()))
            ch_feature_names.extend(list(extract_ar_features(multi_channel_data['eeg'][0, ch, :], channel_info['eeg_fs'], config.EEG_BANDS, config.AR_ORDER).keys()))
            ch_feature_names.extend(list(wavelet_processing(multi_channel_data['eeg'][0, ch, :], config.WAVELET_NAME).keys()))
            ch_feature_names.extend(list(extract_welch_features(multi_channel_data['eeg'][0, ch, :], channel_info['eeg_fs'], config).keys()))
            # Add channel‑specific prefix only to this channel's features
            ch_feature_names = [f"eeg_{ch}_{name}" for name in ch_feature_names]
            feature_names.extend(ch_feature_names)
    else:
        for ch in range(multi_channel_data['eeg'].shape[1]):
            ch_feature_names = []
            ch_feature_names.extend(list(extract_time_domain_features(multi_channel_data['eeg'][0, ch, :]).keys()))
            ch_feature_names.extend(list(extract_ar_features(multi_channel_data['eeg'][0, ch, :], channel_info['eeg_fs'], config.EEG_BANDS, config.AR_ORDER).keys()))
            ch_feature_names.extend(list(wavelet_processing(multi_channel_data['eeg'][0, ch, :], config.WAVELET_NAME).keys()))
            ch_feature_names.extend(list(extract_welch_features(multi_channel_data['eeg'][0, ch, :], channel_info['eeg_fs'], config).keys()))
            # Add channel‑specific prefix only to this channel's features
            ch_feature_names = [f"eeg_{ch}_{name}" for name in ch_feature_names]
            feature_names.extend(ch_feature_names)

        # EOG / EMG feature names stay as they are (no per‑channel EEG prefix)
        feature_names.extend(list(extract_eog_features(multi_channel_data['eog'][0, :, :], channel_info['eog_fs']).keys()))
        feature_names.extend(list(extract_emg_features(multi_channel_data['emg'][0, 0, :], channel_info['emg_fs'], config.EEG_BANDS).keys()))
    return feature_names
