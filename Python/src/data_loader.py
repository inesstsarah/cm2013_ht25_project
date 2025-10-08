import numpy as np
import mne
import xml.etree.ElementTree as ET  # For XML parsing
# TODO: Students may need additional imports:
# import os  # For file handling
# from pathlib import Path  # For path operations


def load_training_data(edf_file_path, xml_file_path):
    """
    STUDENT IMPLEMENTATION AREA: Load EDF and XML files.
    Students must implement actual EDF/XML loading:

    1. Load EDF file using MNE (see read_edf function below)
    2. Load XML annotations (sleep stage labels)
    3. Extract relevant channels (EEG, EOG, EMG)
    4. Segment into 30-second epochs
    5. Handle different sampling rates
    6. Match epochs with sleep stage labels

    Args:
        edf_file_path (str): Path to the EDF file.
        xml_file_path (str): Path to the XML annotation file.

    Returns:
        tuple: A tuple containing:
            - eeg_data (np.ndarray): Shape (n_epochs, n_samples) - EEG data
            - labels (np.ndarray): Shape (n_epochs,) - Sleep stage labels (0-4)
    """
    print(f"Loading training data from {edf_file_path} and {xml_file_path}...")

    # === STEP 1: Load EDF file using MNE ===
    required_channels = ['EEG', 'EOG', 'EMG']
    raw = read_edf(edf_file_path, required_channels)
    # === STEP 2: Load XML annotations ===
    annotations = parse_xml_annotations(xml_file_path)
    # === STEP 3: Extract relevant channels (EEG, EOG, EMG) ===
    raw_multi_channel_data = extract_edf_channels(raw, required_channels)
    # === STEP 4: Segment into 30-second epochs ===
    multi_channel_data = create_30_second_epochs(raw_multi_channel_data)
    # === STEP 5: Handle different sampling rates ===
    # All signal points in the edf file have been interpolated to 125Hz
    # === STEP 6: Match epochs with sleep stage labels ===
    filtered_multi_channel_data, labels = map_annotations_to_epochs(annotations, multi_channel_data)
    # === OPTIONAL STEP 7: Limit to first n_epochs for development/testing ===
    # Realistic size for jumpstart: 2 hours = 2 * 60 * 2 = 240 epochs
    # Real studies are 6-8 hours (720-960 epochs) but 240 is good for development
    n_epochs = 480  # 4 hours of sleep recording for development/testing
    filtered_multi_channel_data, labels = limit_epochs(filtered_multi_channel_data, labels, n_epochs=n_epochs)

    # NOTE FOR STUDENTS: This study has specific multi-channel setup with ACTUAL sampling rates:
    # - 2 EEG channels (C3-A2, C4-A1) at 125 Hz
    # - 2 EOG channels (EOG(L), EOG(R)) at 50 Hz
    # - 1 EMG channel at 125 Hz
    # - ECG at 125 Hz
    # - Other signals: Respiration (10 Hz), SpO2/HR (1 Hz), etc.
    # Students must identify channels by name and handle different sampling rates

    channel_info = {
        'eeg_names': ['C3-A2', 'C4-A1'],
        'eeg_fs': 125,  # Actual sampling rate from study
        'eog_names': ['EOG(L)', 'EOG(R)'],
        'eog_fs': 50,   # Actual sampling rate from study
        'emg_names': ['EMG'],
        'emg_fs': 125,  # Actual sampling rate from study
        'epoch_length': 30
    }
    print(f"Multi-channel structure:")
    print(f"  EEG: {filtered_multi_channel_data['eeg'].shape[0]} epoch total, {filtered_multi_channel_data['eeg'].shape[1]} channels, {filtered_multi_channel_data['eeg'].shape[2]} samples/epoch")
    print(f"  EOG: {filtered_multi_channel_data['eog'].shape[0]} epoch total, {filtered_multi_channel_data['eog'].shape[1]} channels, {filtered_multi_channel_data['eog'].shape[2]} samples/epoch")
    print(f"  EMG: {filtered_multi_channel_data['emg'].shape[0]} epoch total, {filtered_multi_channel_data['emg'].shape[1]} channels, {filtered_multi_channel_data['emg'].shape[2]} samples/epoch")

    return filtered_multi_channel_data, labels, channel_info

def load_holdout_data(edf_file_path):
    """
    STUDENT IMPLEMENTATION AREA: Load holdout EDF files (no labels).

    Similar to load_training_data but without XML annotations.
    Students must implement actual EDF loading for competition data.

    Args:
        edf_file_path (str): Path to the EDF file.

    Returns:
        tuple: (eeg_data, record_info) where:
            - eeg_data (np.ndarray): Shape (n_epochs, n_samples)
            - record_info (dict): Metadata needed for submission (record_id, epoch_count, etc.)
    """
    print(f"Loading holdout data from {edf_file_path}...")

    # TODO: Students must implement:
    # raw = mne.io.read_raw_edf(edf_file_path, preload=True)
    # eeg_channels = raw.pick_channels(['EEG1', 'EEG2', ...])
    # epochs = create_30_second_epochs(eeg_channels)
    # record_info = extract_record_metadata(edf_file_path)

    # DUMMY DATA for jumpstart testing - students must replace:
    print("WARNING: Using dummy data! Students must implement actual EDF loading.")
    n_epochs = 240  # 2 hours for development (real studies: 720-960 epochs)

    # Calculate samples per epoch with CORRECT sampling rates
    eeg_samples = 30 * 125  # 30 seconds at 125 Hz = 3750 samples
    eog_samples = 30 * 50   # 30 seconds at 50 Hz = 1500 samples
    emg_samples = 30 * 125  # 30 seconds at 125 Hz = 3750 samples

    # Multi-channel holdout data with CORRECT sampling rates
    multi_channel_data = {
        'eeg': np.random.randn(n_epochs, 2, eeg_samples),  # 2 EEG channels at 125 Hz
        'eog': np.random.randn(n_epochs, 2, eog_samples),  # 2 EOG channels at 50 Hz
        'emg': np.random.randn(n_epochs, 1, emg_samples),  # 1 EMG channel at 125 Hz
    }

    record_info = {
        'record_id': 1,
        'n_epochs': n_epochs,
        'channels': ['C3-A2', 'C4-A1', 'EOG(L)', 'EOG(R)', 'EMG'],
        'sampling_rates': {'eeg': 125, 'eog': 50, 'emg': 125}
    }
    print(f"Generated dummy multi-channel holdout data: {n_epochs} epochs ({n_epochs/120:.1f} hours)")

    return multi_channel_data, record_info


def read_edf(file_path, required_channels=None):
    """
    EXAMPLE: Read an EDF file using the MNE library.

    This is a basic example. Students should expand this to:
    - Handle different EDF variants
    - Validate channel names and sampling rates
    - Handle missing or corrupted data
    - Extract specific time ranges

    Args:
        file_path (str): The path to the EDF file.
        required_channels (list or None): List of required channel names to validate presence (e.g., ['EEG', 'EOG', 'EMG']).
        tmin (float or None): Start time in seconds for extracting a specific time range.
        tmax (float or None): End time in seconds for extracting a specific time range.

    Returns:
        mne.io.Raw: The raw EDF data.
    """
    try:
        # Load EDF file
        print(f"Parsing EDF file from: {file_path}")
        raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel=None, verbose=False)
        # TODO: Add channel validation, sampling rate checks, etc.
        print(f"Successfully loaded EDF file: {file_path}")
        print(f"Data info:")
        print(f"  Total channels: {len(raw.ch_names)}")
        print(f"  Channel names: {raw.ch_names}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz\n")

        # channel validation
        if required_channels is not None:
            available_channels = raw.ch_names
            print(f"Validating required channels: {required_channels}")
            missing_channels = []
            for ch in required_channels:
                found = any(ch in name.upper() for name in available_channels)
                if not found:
                    missing_channels.append(ch)

            if missing_channels:
                print(f"⚠️ Missing required channels: {missing_channels}\n")
            else:
                print("All required channels are present.\n")

        # Check for missing or corrupted data (e.g., NaNs, extreme values)
        bad_channels = []
        for ch in raw.ch_names:
            data = raw.get_data(picks=[ch])
            if data.size == 0:
                bad_channels.append(f"{ch} contains no data")
            elif np.all(np.isnan(data)):
                bad_channels.append(f"{ch} contains only NaN values")
        if bad_channels:
            for msg in bad_channels:
                print(f"⚠️ {msg}")
        else:
            print("All channels data looks valid.\n")

        return raw
    except Exception as e:
        print(f"Error reading EDF file {file_path}: {e}\n")
        raise


def extract_edf_channels(raw, required_channels):
    """
    Extract specified channel types (EEG, EOG, EMG, ...) from raw EDF data,
    print info, and return numpy arrays.

    Args:
        raw (mne.io.Raw): Raw MNE object
        required_channels (list): List of channel types to extract, e.g., ['EEG','EOG','EMG']

    Returns:
        dict: {'eeg': np.ndarray, 'eog': np.ndarray, 'emg': np.ndarray}
              Each array shape: (n_channels, n_samples)
    """
    try:
        print(f"Extracting channels: {required_channels}...")
        multi_channel_data = {}
        for ch_type in required_channels:
            matched_channels = [ch for ch in raw.ch_names if ch_type.upper() in ch.upper()]
            
            if matched_channels:
                raw_subset = raw.copy().pick(matched_channels)
                data = raw_subset.get_data()  # (n_channels, n_times)
                n_channels, n_times = data.shape
                sfreq = raw_subset.info['sfreq']
                duration_sec = n_times / sfreq
                duration_min = duration_sec / 60
                duration_hr = duration_min / 60

                print(f"{ch_type.upper()}:")
                print(f"  Channels: {n_channels}")
                print(f"  Samples: {n_times}")
                print(f"  Sampling frequency: {sfreq} Hz")
                print(f"  Duration: {duration_sec:.2f} sec ({duration_min:.2f} min, {duration_hr:.2f} hr)")

                # 保存 numpy 数据
                multi_channel_data[ch_type.lower()] = data
            else:
                print(f"⚠️ No {ch_type} channels found in EDF file.")
                multi_channel_data[ch_type.lower()] = None

        print(f"successfully extracted channels: {list(multi_channel_data.keys())}\n")
        return multi_channel_data

    except Exception as e:
        print(f"Error extracting channels in EDF file: {e}")
        raise


def parse_xml_annotations(xml_file_path):
    """
    Students must implement XML parsing for sleep stage annotations.
    The XML format contains sleep stage labels for each epoch.

    Args:
        xml_file_path (str): Path to XML annotation file.

    Returns:
        list: Sleep stage annotations with timestamps.
    """
    try:
        # Parse XML file
        print(f"Parsing XML annotations from: {xml_file_path}")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        annotations = extract_sleep_stages(root)
        return annotations
    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_file_path}")
    except Exception as e:
        print(f"Error Parsing EDF XML annotations {xml_file_path}: {e}\n")
        raise


def extract_sleep_stages(xml_root):
    try:
        # Extract sleep stages and times
        epochs = []
        stages = []
        for event in xml_root.findall('.//ScoredEvent'):
            event_concept = event.find('EventConcept')
            start = event.find('Start')
            duration = event.find('Duration')

            if event_concept is not None and start is not None:
                stage_name = event_concept.text

                # Check if this is a sleep stage event
                # Formats: SDO:WakeState, SDO:NonRapidEyeMovementSleep-N1, SDO:RapidEyeMovementSleep
                # Also support older formats: Wake|0, Stage1|1, etc.
                # Exclude arousal events and other non-stage events
                is_sleep_stage = False
                if 'WakeState' in stage_name or 'RapidEyeMovementSleep' in stage_name or 'NonRapidEyeMovementSleep' in stage_name:
                    is_sleep_stage = True
                elif 'Wake|' in stage_name or 'REM|' in stage_name:
                    is_sleep_stage = True
                elif any(f'Stage{i}' in stage_name for i in range(1, 5)):
                    is_sleep_stage = True
                elif any(f'|{i}' in stage_name for i in range(6)):
                    is_sleep_stage = True

                if is_sleep_stage:
                    start_time = float(start.text)
                    dur = float(duration.text) if duration is not None else 30.0

                    # Map stage names to numeric labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
                    stage_label = None

                    if 'WakeState' in stage_name or stage_name == 'Wake' or 'Wake|0' in stage_name:
                        stage_label = 0
                    elif 'N1' in stage_name or 'Stage1' in stage_name or '|1' in stage_name:
                        stage_label = 1
                    elif 'N2' in stage_name or 'Stage2' in stage_name or '|2' in stage_name:
                        stage_label = 2
                    elif 'N3' in stage_name or 'Stage3' in stage_name or 'Stage4' in stage_name or '|3' in stage_name or '|4' in stage_name:
                        stage_label = 3
                    elif 'RapidEyeMovementSleep' in stage_name or stage_name == 'REM' or '|5' in stage_name:
                        stage_label = 4

                    if stage_label is not None:
                        # Store start time, duration, and stage label
                        epochs.append((start_time, dur, stage_label))

        if not epochs:
            print("Warning: No sleep stage annotations found in XML file")
            print("The XML file may be in a different format or empty")
            return

        # Sort epochs by start time
        epochs = sorted(epochs, key=lambda x: x[0])
        print(f"Extracted {len(epochs)} sleep stage annotations from XML")

        # Extract stages for statistics
        stages = np.array([e[2] for e in epochs])
        total_duration = sum(e[1] for e in epochs)
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        # Print statistics
        print("\nSleep Stage Statistics:")
        print("="*70)
        print(f"Total sleep stage events: {len(stages)}")
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Total duration: {total_duration/3600:.2f} hours")
        print(f"Total epochs: {int(total_duration/30)}")
        print("\nStage distribution:")
        for stage_idx, stage_name in enumerate(stage_names):
            # Count events and calculate total duration for this stage
            stage_events = [e for e in epochs if e[2] == stage_idx]
            count = len(stage_events)
            stage_duration = sum(e[1] for e in stage_events)
            percentage = stage_duration / total_duration * 100
            n_epochs = int(stage_duration / 30)
            print(f"  {stage_name}: {count} events, {n_epochs} epochs ({percentage:.1f}%)")
        print("="*70 + "\n")
        return epochs
    except Exception as e:
        print(f"Error extracting sleep stages from XML: {e}\n")
        raise


def create_30_second_epochs(raw_data, epoch_duration=30):
    """
    Important considerations for students:
    1. Handle different sampling rates for different signal types (ACTUAL rates for this study):
       - EEG (C3-A2, C4-A1): 125 Hz
       - EOG (Left, Right): 50 Hz
       - EMG: 125 Hz
       - ECG: 125 Hz
       - Respiration (Thor/Abdo): 10 Hz
       - SpO2/Heart Rate: 1 Hz

    2. Options for handling multiple sampling rates:
       - Resample all signals to common rate (e.g., 100 Hz)
       - Keep original rates and extract features separately
       - Downsample high-rate signals, upsample low-rate signals

    3. Epoch creation steps:
       - Determine epoch length in samples for each signal type
       - Handle overlapping vs non-overlapping epochs
       - Deal with incomplete final epochs
       - Maintain temporal alignment across signal types

    Args:
        raw_data: MNE Raw object containing multiple channels at different rates.

    Returns:
        dict: Multi-channel epochs with actual channel counts:
            {'eeg': np.ndarray (n_epochs, 2, samples_per_epoch),  # 2 EEG channels
             'eog': np.ndarray (n_epochs, 2, samples_per_epoch),  # 2 EOG channels
             'emg': np.ndarray (n_epochs, 1, samples_per_epoch)}  # 1 EMG channel
    """
    try:
        samples_per_epoch = 125*epoch_duration
        print(f"Creating 30-second epochs from raw data with {samples_per_epoch} samples per epoch...")
        epochs_data = {}
        for signal_name, signal_array in raw_data.items():
            print(f"Processing {signal_name.upper()} data...")
            print(f"  Original shape: {signal_array.shape}")

            n_channels, n_samples = signal_array.shape
            # Integer division for full epochs, discard incomplete final epoch
            n_epochs = n_samples // samples_per_epoch
            # Trim data to full epochs
            trimmed_signal = signal_array[:, :n_epochs * samples_per_epoch]
            # Reshape to (n_epochs, n_channels, samples_per_epoch)
            epochs = trimmed_signal.reshape(n_channels, n_epochs, samples_per_epoch)
            epochs = np.transpose(epochs, (1, 0, 2))  # (n_epochs, n_channels, samples_per_epoch)
            epochs_data[signal_name] = epochs
            print(f"{signal_name.upper()}: {n_epochs} epochs, {n_channels} channels, {samples_per_epoch} samples/epoch")
        print(f"Successfully created epochs for signals: {list(epochs_data.keys())}\n")
        return epochs_data
    except Exception as e:
        print(f"Error segmenting data into epochs: {e}\n")
        raise


def map_annotations_to_epochs(annotations, epochs):
    """
    Map sleep stage annotations to epochs.
    Args:
        annotations (list): List of (start_time, duration, stage_label) tuples.
        epochs (dict): Multi-channel epochs with shape info.
    Returns:
        np.ndarray: Sleep stage labels for each epoch.
    """
    try:
        print("Mapping annotations to epochs...")
        n_epochs = epochs['eeg'].shape[0]  # Assuming EEG defines epoch count
        epoch_duration = 30
        labels = np.full(n_epochs, -1)  # Initialize with -1 (unknown)
        
        for start_time, duration, stage_label in annotations:
            start_epoch = int(start_time // epoch_duration)
            end_epoch = int((start_time + duration) // epoch_duration)

            for epoch_idx in range(start_epoch, min(end_epoch, n_epochs)):
                labels[epoch_idx] = stage_label

        # Identify epochs with valid labels
        valid_mask = labels != -1
        # Filter epochs for each channel
        filtered_epochs = {}
        for key, data in epochs.items():
            filtered_epochs[key] = data[valid_mask]
        # Filter labels
        labels = labels[valid_mask]

        removed_epochs = np.where(~valid_mask)[0]
        if removed_epochs.size > 0:
            print(f"Removed {removed_epochs.size} epochs with unknown labels: {removed_epochs.tolist()}")

        print(f"Final epoch count after mapping: {labels.shape[0]} epochs with labels")
        return filtered_epochs, labels
    
    except Exception as e:
        print(f"Error mapping annotations to epochs: {e}\n")
        raise


def limit_epochs(epochs, labels, n_epochs=240):
    """
    Limit the dataset to the first n_epochs.

    Args:
        epochs (dict): Multi-channel epoch data.
        labels (np.ndarray): Array of sleep stage labels for each epoch.
        n_epochs (int): Number of epochs to keep.

    Returns:
        (filtered_epochs, filtered_labels): A tuple containing the limited epochs and corresponding labels.
    """
    try:
        total_epochs = labels.shape[0]

        # Check if the requested number exceeds available epochs
        if n_epochs > total_epochs:
            print(f"⚠️ Warning: n_epochs={n_epochs} exceeds available epochs ({total_epochs}). Using all available epochs.")
            n_epochs = total_epochs

        # If the input is a dictionary (multi-channel data)
        if isinstance(epochs, dict):
            filtered_epochs = {}
            for ch_name, data in epochs.items():
                filtered_epochs[ch_name] = data[:n_epochs]
        else:
            TypeError("Epochs input must be a dictionary of multi-channel data.")

        # Truncate labels to match the selected number of epochs
        filtered_labels = labels[:n_epochs]

        print(f"Using first {n_epochs} epochs out of {total_epochs}.")
        return filtered_epochs, filtered_labels
    except Exception as e:
        print(f"Error limiting epochs: {e}\n")
        raise
