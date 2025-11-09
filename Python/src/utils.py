import os
import joblib
import numpy as np

def save_cache(data, filename, cache_dir):
    """
    Saves data to a cache file.

    Args:
        data (any): The data to be cached.
        filename (str): The name of the cache file.
        cache_dir (str): The directory to save the cache file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)
    joblib.dump(data, filepath)
    print(f"Data cached to {filepath}")

def load_cache(filename, cache_dir):
    """
    Loads data from a cache file.

    Args:
        filename (str): The name of the cache file.
        cache_dir (str): The directory where the cache file is located.

    Returns:
        any: The loaded data, or None if the file does not exist.
    """
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        print(f"Loading data from cache: {filepath}")
        return joblib.load(filepath)
    print(f"Cache file not found: {filepath}")
    return None


# Clinical plausibility check
def calculate_sleep_metrics(labels: np.ndarray, epoch_duration: int = 30) -> dict:
    """
    Calculate sleep architecture metrics from epoch labels.

    Args:
        labels: array of sleep stage labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        epoch_duration: seconds per epoch (default 30)

    Returns:
        metrics: dict of sleep architecture values
    """
    if len(labels) == 0:
        return {}
    
    # Convert to numpy array for easier manipulation
    labels = np.array(labels)
    n_epochs = len(labels)
    total_time_minutes = n_epochs * epoch_duration / 60  # Total time in bed (minutes)
    
    metrics = {}
    
    # 1. Find sleep onset (first non-wake epoch)
    sleep_epochs = np.where(labels != 0)[0]  # Non-wake epochs
    if len(sleep_epochs) == 0:
        # No sleep detected
        metrics['sleep_onset_latency'] = None
        metrics['total_sleep_time'] = 0
        metrics['sleep_efficiency'] = 0
        metrics['wake_after_sleep_onset'] = 0
        metrics['rem_latency'] = None
        metrics['n_awakenings'] = 0
        metrics['stage_percentages'] = {'N1': 0, 'N2': 0, 'N3': 0, 'REM': 0}
        metrics['rem_cycles'] = 0
        return metrics
    
    first_sleep_epoch = sleep_epochs[0]
    sleep_onset_latency = first_sleep_epoch * epoch_duration / 60  # minutes
    
    # 2. Calculate Total Sleep Time (TST) - sum of all sleep epochs
    sleep_epochs_mask = labels != 0  # Non-wake epochs
    total_sleep_epochs = np.sum(sleep_epochs_mask)
    total_sleep_time = total_sleep_epochs * epoch_duration / 60  # minutes
    
    # 3. Sleep Efficiency = (TST / Time in Bed) × 100%
    sleep_efficiency = (total_sleep_time / total_time_minutes) * 100 if total_time_minutes > 0 else 0
    
    # 4. Wake After Sleep Onset (WASO) - wake epochs after first sleep
    wake_after_sleep = labels[first_sleep_epoch:]
    waso_epochs = np.sum(wake_after_sleep == 0)
    wake_after_sleep_onset = waso_epochs * epoch_duration / 60  # minutes
    
    # 5. REM Latency - time from sleep onset to first REM
    rem_epochs = np.where(labels == 4)[0]  # REM epochs
    if len(rem_epochs) == 0:
        rem_latency = None
    else:
        first_rem_epoch = rem_epochs[0]
        if first_rem_epoch >= first_sleep_epoch:
            rem_latency = (first_rem_epoch - first_sleep_epoch) * epoch_duration / 60  # minutes
        else:
            rem_latency = None
    
    # 6. Number of Awakenings - count wake periods after sleep onset
    wake_after_sleep_binary = (wake_after_sleep == 0).astype(int)
    # Count transitions from sleep (0) to wake (1)
    awakenings = 0
    if len(wake_after_sleep_binary) > 1:
        for i in range(1, len(wake_after_sleep_binary)):
            if wake_after_sleep_binary[i-1] == 0 and wake_after_sleep_binary[i] == 1:
                awakenings += 1
    
    # 7. Sleep Stage Percentages (relative to TST)
    stage_counts = np.bincount(labels[labels != 0], minlength=5)  # Count non-wake stages
    stage_percentages = {}
    if total_sleep_epochs > 0:
        stage_percentages = {
            'N1': (stage_counts[1] / total_sleep_epochs) * 100,
            'N2': (stage_counts[2] / total_sleep_epochs) * 100,
            'N3': (stage_counts[3] / total_sleep_epochs) * 100,
            'REM': (stage_counts[4] / total_sleep_epochs) * 100
        }
    else:
        stage_percentages = {'N1': 0, 'N2': 0, 'N3': 0, 'REM': 0}
    
    # 8. REM Cycle Count and Duration
    rem_cycles = 0
    if len(rem_epochs) > 0:
        # Find consecutive REM periods
        rem_binary = np.zeros(len(labels), dtype=int)
        rem_binary[rem_epochs] = 1
        rem_duration = np.sum(rem_binary) * epoch_duration / 60  # minutes
        # Count transitions from non-REM to REM
        for i in range(1, len(rem_binary)):
            if rem_binary[i-1] == 0 and rem_binary[i] == 1:
                rem_cycles += 1
    
    # Store all metrics
    metrics = {
        'SOL': sleep_onset_latency,  # minutes
        'TST': total_sleep_time,  # minutes
        'SE': sleep_efficiency,  # percentage
        'WASO': wake_after_sleep_onset,  # minutes
        'REM_latency': rem_latency,  # minutes
        'n_awakenings': awakenings,
        'sleep_stage_percentages': stage_percentages,
        'REM_cycles': rem_cycles,
        'REM_duration': rem_duration,  # minutes
        'total_time_in_bed': total_time_minutes,  # minutes
        'n_epochs': n_epochs
    }
    
    return metrics