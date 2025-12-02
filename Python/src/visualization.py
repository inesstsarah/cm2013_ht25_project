import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xml.etree.ElementTree as ET
import os
import matplotlib.patches as mpatches
from src.utils import calculate_sleep_metrics
from scipy.stats import ttest_rel

# Try to import MNE for EDF reading (more lenient than pyedflib)
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    try:
        import pyedflib
        HAS_PYEDFLIB = True
    except ImportError:
        HAS_PYEDFLIB = False


def plot_sample_epoch(edf_path, epoch_idx=0, epoch_duration=30):
    """
    Plot all signals from a sample epoch in an EDF file.

    Args:
        edf_path (str): Path to the EDF file.
        epoch_idx (int): Index of the epoch to plot (default: 0).
        epoch_duration (int): Duration of each epoch in seconds (default: 30).
    """
    if not HAS_MNE and not HAS_PYEDFLIB:
        print("Error: Neither MNE nor pyedflib is installed.")
        print("Please install one: pip install mne  OR  pip install pyedflib")
        return

    try:
        # Reset matplotlib to defaults
        import matplotlib
        matplotlib.rcdefaults()

        # Calculate epoch boundaries
        start_time = epoch_idx * epoch_duration

        if HAS_MNE:
            # Use MNE (more lenient with EDF format issues)
            raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)

            n_channels = len(raw.ch_names)
            channel_labels = raw.ch_names

            # Extract data for this epoch
            start_sample = int(start_time * raw.info['sfreq'])
            stop_sample = int((start_time + epoch_duration) * raw.info['sfreq'])

            # Convert from Volts to microvolts for better visualization
            # MNE loads data in Volts by default
            data_all = raw[:, start_sample:stop_sample][0] * 1e6
            times = np.arange(data_all.shape[1]) / raw.info['sfreq'] + start_time

        else:
            # Fallback to pyedflib
            with pyedflib.EdfReader(edf_path) as edf:
                n_channels = edf.signals_in_file
                channel_labels = edf.getSignalLabels()
                sampling_freqs = [edf.getSampleFrequency(i) for i in range(n_channels)]

                data_all = []
                for ch_idx in range(n_channels):
                    fs = sampling_freqs[ch_idx]
                    start_sample = int(start_time * fs)
                    n_samples = int(epoch_duration * fs)
                    signal = edf.readSignal(ch_idx, start=start_sample, n=n_samples)
                    data_all.append(signal)

                # Create time axis
                max_samples = max(len(d) for d in data_all)
                times = np.linspace(start_time, start_time + epoch_duration, max_samples)

        # Create subplots - EXACTLY like the diagnostic plot that worked
        fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2*n_channels),
                                facecolor='white', edgecolor='black')
        if n_channels == 1:
            axes = [axes]

        print(f"\nPlotting Epoch {epoch_idx} (Time: {start_time}-{start_time+epoch_duration}s)")
        print("="*70)

        for ch_idx in range(n_channels):
            label = channel_labels[ch_idx]

            if HAS_MNE:
                signal = data_all[ch_idx]
            else:
                signal = data_all[ch_idx]

            # Set white background for subplot - EXACTLY like diagnostic
            axes[ch_idx].set_facecolor('white')

            # Plot with VERY visible settings - EXACTLY like diagnostic
            axes[ch_idx].plot(times, signal, 'b-', linewidth=2.0, solid_capstyle='round')

            # Add unit to ylabel for bio-signal channels
            if 'EEG' in label or 'EOG' in label or 'EMG' in label or 'ECG' in label:
                ylabel = f'{label} (µV)'
            else:
                ylabel = f'{label}'

            axes[ch_idx].set_ylabel(ylabel, fontsize=11, fontweight='bold')
            axes[ch_idx].grid(True, color='gray', alpha=0.4, linestyle='-', linewidth=0.5)
            axes[ch_idx].set_xlim(times[0], times[-1])

            # Explicit y-limits - EXACTLY like diagnostic
            y_margin = (signal.max() - signal.min()) * 0.15
            axes[ch_idx].set_ylim(signal.min() - y_margin, signal.max() + y_margin)

            # Add text showing we have data - EXACTLY like diagnostic
            axes[ch_idx].text(0.98, 0.95, f'n={len(signal)}', transform=axes[ch_idx].transAxes,
                            ha='right', va='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

            print(f"  {label}: {len(signal)} samples, range=[{signal.min():.1f}, {signal.max():.1f}]")

        axes[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Sleep Signals - Epoch {epoch_idx} ({epoch_duration}s window)',
                         fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save figure explicitly before showing
        output_path = f"epoch{epoch_idx}_signals.png"
        plt.savefig(output_path, dpi=100, facecolor='white', edgecolor='black', bbox_inches='tight')
        print(f"\n✓ Saved to {output_path}")

        plt.show()

    except FileNotFoundError:
        print(f"Error: EDF file not found at {edf_path}")
    except Exception as e:
        print(f"Error reading EDF file: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_hypnogram(xml_path, edf_path=None):
    """
    Plot hypnogram (sleep stage progression) from XML annotations.

    Args:
        xml_path (str): Path to the XML annotation file.
        edf_path (str, optional): Path to EDF file to get recording duration.
    """
    try:
        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract sleep stages and times
        epochs = []
        stages = []

        # Try different XML structures (Compumedics format)
        for event in root.findall('.//ScoredEvent'):
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

        # Create hypnogram plot
        fig, ax = plt.subplots(figsize=(15, 5))

        # Plot as step function
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        stage_colors = ['red', 'orange', 'green', 'blue', 'purple']

        # Create step plot - each event has (start_time, duration, stage_label)
        for start_time, duration, stage_label in epochs:
            start_epoch = start_time / 30.0
            end_epoch = (start_time + duration) / 30.0
            ax.hlines(stage_label, start_epoch, end_epoch,
                     colors=stage_colors[int(stage_label)], linewidth=2)

        # Extract stages for statistics
        stages = np.array([e[2] for e in epochs])
        total_duration = sum(e[1] for e in epochs)

        # Styling
        ax.set_yticks(range(5))
        ax.set_yticklabels(stage_names)
        ax.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch Number (30s epochs)', fontsize=12, fontweight='bold')
        ax.set_title('Hypnogram - Sleep Stage Progression', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_ylim(-0.5, 4.5)

        # Add time axis on top
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        # Convert epochs to hours
        max_epoch = (epochs[-1][0] + epochs[-1][1]) / 30.0  # Last event end time
        hour_ticks = np.arange(0, max_epoch, 120)  # 120 epochs = 1 hour
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels([f'{h/120:.1f}' for h in hour_ticks])

        plt.tight_layout()
        plt.show()

        # Print statistics
        print("\nSleep Stage Statistics:")
        print("="*70)
        print(f"Total sleep stage events: {len(stages)}")
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

    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_path}")
    except Exception as e:
        print(f"Error reading XML file: {str(e)}")
        import traceback
        traceback.print_exc()


def visualize_results(results, record_ids, config):
    """
    Visualizes the results of the classification.

    Args:
        model (object): The trained model.
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.
    """
    print("Visualizing results...")
    # TODO: Add more visualizations as needed (e.g., feature importance).
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    # visualize confusion matrix
    print("Visualizing confusion matrix...")
    _plot_confusion_matrix(results['y_true_aggregate'], results['y_pred_aggregate'], class_names, config)

    # visualize hypnograms side by side
    print("Visualizing hypnograms side by side...")
    _visualize_sidebyside_hypnograms(results['y_true_aggregate'], results['y_pred_aggregate'], record_ids, config)

    # visualize stage percentage comparison
    print("Visualizing stage percentage comparison...")
    _visualize_stage_percentage_comparison(results['y_true_aggregate'], results['y_pred_aggregate'], config, record_ids)

    # Calculate sleep metrics once for all subjects
    print("\nCalculating sleep architecture metrics...")
    sleep_metrics_data = _calculate_all_sleep_metrics(results['y_true_aggregate'], results['y_pred_aggregate'], record_ids, config)
    
    # sleep architecture metrics (Bland-Altman plots, correlation analysis)
    print("\nVisualizing sleep architecture metrics...")
    _plot_from_sleep_metrics(sleep_metrics_data, config)


def visualize_fft(signal: np.ndarray, fs: float, ax: plt.Axes = None, title: str = "FFT of Signal") -> None:
    """
    Plot the FFT of a signal on a given matplotlib Axes.

    Args:
        signal (np.ndarray): The input signal.
        fs (float): Sampling frequency (Hz).
        ax (matplotlib.axes.Axes, optional): Axes to plot on. 
                                             If None, create a new figure.
        title (str, optional): Plot title.
    """
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/fs)
    fft_values = np.fft.fft(signal)
    magnitude = np.abs(fft_values)[:n // 2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(freq[:n // 2], magnitude)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)


def visualize_signal(signal: np.ndarray, fs: float, ax: plt.Axes = None, title: str = "Time-domain Signal") -> None:
    """
    Plot the time-domain waveform of a signal on a given matplotlib Axes.

    Args:
        signal (np.ndarray): The input signal.
        fs (float): Sampling frequency (Hz).
        ax (matplotlib.axes.Axes, optional): Axes to plot on. 
                                             If None, create a new figure.
        title (str, optional): Plot title.
    """
    n = len(signal)
    t = np.arange(n) / fs  

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, signal*1e6)  # Convert to microvolts for better visualization
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_ylim(np.min(signal)*1.1e6, np.max(signal)*1.1e6)
    ax.grid(True)


def _calculate_all_sleep_metrics(y_true: np.ndarray, y_pred: np.ndarray, record_ids: np.ndarray, config: dict, epoch_duration: int = 30) -> dict:
    """
    Calculate sleep metrics for all subjects once.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        record_ids (np.ndarray): Record identifiers
        config (module): Configuration module
        epoch_duration (int): Duration of each epoch in seconds
        
    Returns:
        dict: Sleep metrics data for all subjects
    """

    unique_records = np.unique(record_ids)
    all_true_metrics = []
    all_pred_metrics = []
    
    for record_id in unique_records:
        indices = np.where(record_ids == record_id)[0]
        if len(indices) == 0:
            continue
            
        true_labels = y_true[indices]
        pred_labels = y_pred[indices]
        
        true_metrics = calculate_sleep_metrics(true_labels, epoch_duration)
        pred_metrics = calculate_sleep_metrics(pred_labels, epoch_duration)
        
        all_true_metrics.append(true_metrics)
        all_pred_metrics.append(pred_metrics)
    
    return {
        'all_true_metrics': all_true_metrics,
        'all_pred_metrics': all_pred_metrics
    }


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, config: dict) -> None:
    """
    Plots a confusion matrix.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        class_names (list): The names of the classes.
    """
    plt.figure(figsize=(20, 20))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.title("Confusion Matrix")
    # plt.show(block=False)
    save_path = os.path.join(config.FIGURES_CLASSIFICATION_DIR, "confusion matrix.png")
    plt.gcf().savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(plt.gcf()) # release memory


def _visualize_sidebyside_hypnograms(y_true: np.ndarray, y_pred: np.ndarray, record_ids: np.ndarray, config: dict) -> None:
    # Plot as step function
    # find ids
    unique_records = np.unique(record_ids)

    # Reuse a single figure for all records to reduce overhead
    fig = plt.figure(figsize=(15, 5))

    for target_id in unique_records:
        print(f"Visualizing hypnogram for record {target_id}...")
        # Clear previous content and create a fresh Axes on the same figure
        fig.clf()
        ax = fig.add_subplot(111)

        indices = np.where(record_ids == target_id)[0]
        nepochs = len(indices)
        stage_label_list_true = y_true[indices]
        stage_label_list_pred = y_pred[indices]

        # Create step plot - ground truth
        x = np.arange(nepochs)
        offset = 0.3 
        # Plot ground truth as scatter
        true_colors = np.asarray(config.STAGE_COLORS)[stage_label_list_true.astype(int)] # color mapping
        ax.scatter(x, stage_label_list_true + offset, c=true_colors, s=2, marker='s', alpha=1.0, label='Ground Truth')
        
        # Prediction
        pred_colors = np.asarray(config.LIGHT_STAGE_COLORS)[stage_label_list_pred.astype(int)]
        ax.scatter(x, stage_label_list_pred - offset, c=pred_colors, s=2, marker='s', alpha=0.4, label='Prediction')

        # Styling
        ax.set_yticks(range(5))
        ax.set_yticklabels(config.STAGE_NAMES)
        ax.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch Number (30s epochs)', fontsize=12, fontweight='bold')
        ax.set_title(f'Side-by-Side Hypnogram - {target_id}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_ylim(-0.5, 4.5)
        patch_truth = mpatches.Patch(color=config.STAGE_COLORS[0], label='Ground Truth')
        patch_pred  = mpatches.Patch(color=config.LIGHT_STAGE_COLORS[0], alpha=0.4, label='Prediction')
        ax.legend(handles=[patch_truth, patch_pred],
                loc='upper right',
                frameon=False)

        # Add time axis on top
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        # Convert epochs to hours
        max_epoch = nepochs
        hour_ticks = np.arange(0, max_epoch, 120)  # 120 epochs = 1 hour
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels([f'{h/120:.1f}' for h in hour_ticks])

        fig.tight_layout()
        save_path = os.path.join(config.FIGURES_CLASSIFICATION_DIR, f"sidebyde_hypnograms_{target_id}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved hypnogram to {save_path}")
    
    plt.close(fig)  # release memory for the reused figure after all saves


def _compute_percentages(labels: np.ndarray, config: dict) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=len(config.STAGE_NAMES))
    total = counts.sum()
    if total == 0:
        return np.zeros(len(config.STAGE_NAMES), dtype=float)
    return counts / total * 100.0


def _print_stage_percentage_table(y_true: np.ndarray, y_pred: np.ndarray, config: dict, record_id: str = None) -> None:
    """
    Print a table showing stage percentage comparison with error column.
    
    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predicted labels.
        config (module): Config module providing STAGE_NAMES.
        record_id (str | None): Optional record identifier for table title.
    """

    true_pct = _compute_percentages(y_true, config)
    pred_pct = _compute_percentages(y_pred, config)
    error = np.abs(true_pct - pred_pct)
    
    title = f"Stage Percentage Comparison - {record_id}" if record_id else "Stage Percentage Comparison (Overall)"
    print(f"\n{title}")
    print("=" * 60)
    print(f"{'Stage':<8} {'Ground Truth':<12} {' Predicted':<12} {'Error':<8}")
    print("-" * 60)
    
    for i, stage in enumerate(config.STAGE_NAMES):
        print(f"{stage:<8} {true_pct[i]:>10.1f}% {pred_pct[i]:>10.1f}% {error[i]:>6.1f}%")
    
    print("-" * 60)
    print(f"{'Total':<8} {true_pct.sum():>10.1f}% {pred_pct.sum():>10.1f}% {error.sum():>6.1f}%")
    print("=" * 60)


def _visualize_stage_percentage_comparison(y_true: np.ndarray, y_pred: np.ndarray, config: dict, record_ids: np.ndarray = None) -> None:
    """
    Plot bar charts comparing sleep stage percentage distributions (Ground Truth vs Prediction).

    Args:
        y_true (np.ndarray): Ground-truth labels, shape (N,), values in [0..4].
        y_pred (np.ndarray): Predicted labels, shape (N,), values in [0..4].
        config (module): Config module providing FIGURES_CLASSIFICATION_DIR, STAGE_NAMES,
                         STAGE_COLORS, LIGHT_STAGE_COLORS.
        record_ids (np.ndarray | None): Optional, shape (N,). If provided, save one figure per
                                        unique record id; otherwise save a single overall figure.
    """

    # Reuse a single figure to reduce overhead
    fig = plt.figure(figsize=(10, 6))

    if record_ids is None:
        print("Visualizing stage percentage comparison for overall...")
        fig.clf()
        ax = fig.add_subplot(111)

        true_pct = _compute_percentages(y_true, config)
        pred_pct = _compute_percentages(y_pred, config)

        x = np.arange(len(config.STAGE_NAMES))
        width = 0.38

        bars_true = ax.bar(
            x - width/2,
            true_pct,
            width,
            label='Ground Truth',
            color=config.STAGE_COLORS,
            alpha=0.9
        )
        bars_pred = ax.bar(
            x + width/2,
            pred_pct,
            width,
            label='Prediction',
            color=config.LIGHT_STAGE_COLORS,
            alpha=0.7
        )

        ax.set_xticks(x)
        ax.set_xticklabels(config.STAGE_NAMES)
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Stage Percentage Comparison (Overall)', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()

        # Add numeric labels on the top of bars
        ax.bar_label(bars_true, fmt='%.1f%%', padding=3)
        ax.bar_label(bars_pred, fmt='%.1f%%', padding=3)

        fig.tight_layout()
        save_path = os.path.join(config.FIGURES_CLASSIFICATION_DIR, 'stage_percentage_comparison_overall.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Print table for overall comparison
        _print_stage_percentage_table(y_true, y_pred, config)
    else:
        unique_records = np.unique(record_ids)

        for rid in unique_records:
            print(f"\nVisualizing stage percentage comparison for record {rid}...")
            fig.clf()
            ax = fig.add_subplot(111)

            idx = np.where(record_ids == rid)[0]
            if idx.size == 0:
                continue

            true_pct = _compute_percentages(y_true[idx], config)
            pred_pct = _compute_percentages(y_pred[idx], config)

            x = np.arange(len(config.STAGE_NAMES))
            width = 0.38

            bars_true = ax.bar(
                x - width/2,
                true_pct,
                width,
                label='Ground Truth',
                color=config.STAGE_COLORS,
                alpha=0.9
            )
            bars_pred = ax.bar(
                x + width/2,
                pred_pct,
                width,
                label='Prediction',
                color=config.LIGHT_STAGE_COLORS,
                alpha=0.7
            )

            ax.set_xticks(x)
            ax.set_xticklabels(config.STAGE_NAMES)
            ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'Stage Percentage Comparison - {rid}', fontsize=14, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend()

            ax.bar_label(bars_true, fmt='%.1f%%', padding=3)
            ax.bar_label(bars_pred, fmt='%.1f%%', padding=3)

            fig.tight_layout()
            save_path = os.path.join(config.FIGURES_CLASSIFICATION_DIR, f'stage_percentage_comparison_{rid}.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Print table for this record
            _print_stage_percentage_table(y_true[idx], y_pred[idx], config, record_id=rid)

    plt.close(fig)


def _create_bland_altman_plots(all_true_metrics: list, all_pred_metrics: list, metric_keys: list, metric_names: dict, metric_units: dict, config: dict) -> None:
    """
    Create Bland-Altman plots with multiple subjects (each subject is one data point).
    
    Args:
        all_true_metrics (list): List of true metrics for each subject
        all_pred_metrics (list): List of predicted metrics for each subject
        metric_keys (list): List of metric keys to plot
        metric_names (dict): Display names for metrics
        metric_units (dict): Units for metrics
        config (module): Configuration module
    """
    # Filter out metrics with None values across all subjects
    valid_metrics = []
    for key in metric_keys:
        valid_count = 0
        for true_metrics, pred_metrics in zip(all_true_metrics, all_pred_metrics):
            if (key in true_metrics and key in pred_metrics and 
                true_metrics[key] is not None and pred_metrics[key] is not None):
                valid_count += 1
        
        if valid_count >= 3:  # Need at least 3 subjects for meaningful statistics
            valid_metrics.append(key)
    
    if not valid_metrics:
        print("No valid metrics available for Bland-Altman plot (need at least 3 subjects)")
        return
    
    print("Creating Bland-Altman plots...")
    # Calculate number of subplots needed
    n_metrics = len(valid_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, metric_key in enumerate(valid_metrics):
        ax = axes[i]
        
        # Collect data points from all subjects
        true_values = []
        pred_values = []
        
        for true_metrics, pred_metrics in zip(all_true_metrics, all_pred_metrics):
            if (metric_key in true_metrics and metric_key in pred_metrics and 
                true_metrics[metric_key] is not None and pred_metrics[metric_key] is not None):
                true_values.append(true_metrics[metric_key])
                pred_values.append(pred_metrics[metric_key])
        
        if len(true_values) < 3:
            continue
            
        true_values = np.array(true_values)
        pred_values = np.array(pred_values)
        
        # Calculate Bland-Altman statistics
        mean_values = (true_values + pred_values) / 2
        diff_values = true_values - pred_values
        
        # Plot the data points (each subject is one point)
        ax.scatter(mean_values, diff_values, alpha=0.7, s=60, c='blue', edgecolors='black')
        
        # Add subject labels
        for j, (mean_val, diff_val) in enumerate(zip(mean_values, diff_values)):
            ax.annotate(f'S{j+1}', (mean_val, diff_val), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Calculate and plot bias (mean difference)
        bias = np.mean(diff_values)
        ax.axhline(y=bias, color='red', linestyle='--', linewidth=2, label=f'Bias: {bias:.2f}')
        
        # Calculate and plot limits of agreement (±1.96 * std)
        std_diff = np.std(diff_values, ddof=1)  # Use sample standard deviation
        upper_limit = bias + 1.96 * std_diff
        lower_limit = bias - 1.96 * std_diff
        
        ax.axhline(y=upper_limit, color='red', linestyle=':', alpha=0.7, 
                  label=f'Upper LoA: {upper_limit:.2f}')
        ax.axhline(y=lower_limit, color='red', linestyle=':', alpha=0.7,
                  label=f'Lower LoA: {lower_limit:.2f}')
        
        # Styling
        ax.set_xlabel(f'Mean of True and Predicted ({metric_units[metric_key]})', fontsize=10)
        ax.set_ylabel(f'Difference (True - Predicted) ({metric_units[metric_key]})', fontsize=10)
        ax.set_title(f'{metric_names[metric_key]} - All Subjects', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add text with statistics
        stats_text = f'Bias: {bias:.2f}\nLoA: ±{1.96*std_diff:.2f}\nN={len(true_values)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if config and hasattr(config, 'FIGURES_CLASSIFICATION_DIR'):
        save_path = os.path.join(config.FIGURES_CLASSIFICATION_DIR, 'bland_altman_all_subjects.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Bland-Altman plot to {save_path}\n")
    
    plt.close(fig)


def _create_correlation_plots(all_true_metrics: list, all_pred_metrics: list, metric_keys: list, metric_names: dict, metric_units: dict, config: dict) -> None:
    """
    Create correlation plots with multiple subjects (each subject is one data point).
    
    Args:
        all_true_metrics (list): List of true metrics for each subject
        all_pred_metrics (list): List of predicted metrics for each subject
        metric_keys (list): List of metric keys to plot
        metric_names (dict): Display names for metrics
        metric_units (dict): Units for metrics
        config (module): Configuration module
    """
    # Filter out metrics with None values across all subjects
    valid_metrics = []
    for key in metric_keys:
        valid_count = 0
        for true_metrics, pred_metrics in zip(all_true_metrics, all_pred_metrics):
            if (key in true_metrics and key in pred_metrics and 
                true_metrics[key] is not None and pred_metrics[key] is not None):
                valid_count += 1
        
        if valid_count >= 3:  # Need at least 3 subjects for meaningful correlation
            valid_metrics.append(key)
    
    if not valid_metrics:
        print("No valid metrics available for correlation plot (need at least 3 subjects)")
        return
    
    print("Creating Correlation plots...")
    # Calculate number of subplots needed
    n_metrics = len(valid_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, metric_key in enumerate(valid_metrics):
        ax = axes[i]
        
        # Collect data points from all subjects
        true_values = []
        pred_values = []
        
        for true_metrics, pred_metrics in zip(all_true_metrics, all_pred_metrics):
            if (metric_key in true_metrics and metric_key in pred_metrics and 
                true_metrics[metric_key] is not None and pred_metrics[metric_key] is not None):
                true_values.append(true_metrics[metric_key])
                pred_values.append(pred_metrics[metric_key])
        
        if len(true_values) < 3:
            continue
            
        true_values = np.array(true_values)
        pred_values = np.array(pred_values)
        
        # Plot scatter plot
        ax.scatter(true_values, pred_values, alpha=0.7, s=60, c='blue', edgecolors='black')
        
        # Add subject labels
        for j, (true_val, pred_val) in enumerate(zip(true_values, pred_values)):
            ax.annotate(f'S{j+1}', (true_val, pred_val), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.7)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(true_values, pred_values)[0, 1]
        
        # Calculate R-squared
        r_squared = correlation ** 2
        
        # Fit linear regression line
        z = np.polyfit(true_values, pred_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(true_values.min(), true_values.max(), 100)
        y_line = p(x_line)
        ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=2, label='Regression Line')
        
        # Perfect agreement line (y = x)
        min_val = min(true_values.min(), pred_values.min())
        max_val = max(true_values.max(), pred_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='Perfect Agreement')
        
        # Styling
        ax.set_xlabel(f'True {metric_names[metric_key]} ({metric_units[metric_key]})', fontsize=10)
        ax.set_ylabel(f'Predicted {metric_names[metric_key]} ({metric_units[metric_key]})', fontsize=10)
        ax.set_title(f'{metric_names[metric_key]} - Correlation Analysis', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add correlation statistics
        stats_text = f'r = {correlation:.3f}\nR² = {r_squared:.3f}\nN = {len(true_values)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend
        ax.legend(fontsize=8, loc='lower right')
        
        # Set equal aspect ratio for better visualization
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if config and hasattr(config, 'FIGURES_CLASSIFICATION_DIR'):
        save_path = os.path.join(config.FIGURES_CLASSIFICATION_DIR, 'correlation_analysis.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation plot to {save_path}\n")
    
    plt.close(fig)


def _create_distribution_plots(all_true_metrics: list, all_pred_metrics: list, metric_keys: list, metric_names: dict, metric_units: dict, config: dict) -> None:
    """
    Create distribution plots showing metric variability across subjects.
    
    Args:
        all_true_metrics (list): List of true metrics for each subject
        all_pred_metrics (list): List of predicted metrics for each subject
        metric_keys (list): List of metric keys to plot
        metric_names (dict): Display names for metrics
        metric_units (dict): Units for metrics
        config (module): Configuration module
    """
    # Filter out metrics with None values across all subjects
    valid_metrics = []
    for key in metric_keys:
        valid_count = 0
        for true_metrics, pred_metrics in zip(all_true_metrics, all_pred_metrics):
            if (key in true_metrics and key in pred_metrics and 
                true_metrics[key] is not None and pred_metrics[key] is not None):
                valid_count += 1
        
        if valid_count >= 3:  # Need at least 3 subjects for meaningful distribution
            valid_metrics.append(key)
    
    if not valid_metrics:
        print("No valid metrics available for distribution plot (need at least 3 subjects)")
        return
    
    print("Creating Distribution plots...")
    # Calculate number of subplots needed
    n_metrics = len(valid_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, metric_key in enumerate(valid_metrics):
        ax = axes[i]
        
        # Collect data points from all subjects
        true_values = []
        pred_values = []
        
        for true_metrics, pred_metrics in zip(all_true_metrics, all_pred_metrics):
            if (metric_key in true_metrics and metric_key in pred_metrics and 
                true_metrics[metric_key] is not None and pred_metrics[metric_key] is not None):
                true_values.append(true_metrics[metric_key])
                pred_values.append(pred_metrics[metric_key])
        
        if len(true_values) < 3:
            continue
            
        true_values = np.array(true_values)
        pred_values = np.array(pred_values)
        
        # Create box plot
        data_to_plot = [true_values, pred_values]
        box_plot = ax.boxplot(data_to_plot, labels=['Ground Truth', 'Predicted'], 
                             patch_artist=True, showmeans=True, meanline=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style the median lines
        for median in box_plot['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Style the mean lines
        for mean in box_plot['means']:
            mean.set_color('red')
            mean.set_linewidth(2)
            mean.set_linestyle('--')
        
        # Add individual data points
        n_subjects = len(true_values)
        x_positions = np.random.normal(1, 0.04, n_subjects)  # Jitter for true values
        ax.scatter(x_positions, true_values, alpha=0.6, s=30, c='blue', edgecolors='black', linewidth=0.5)
        
        x_positions = np.random.normal(2, 0.04, n_subjects)  # Jitter for predicted values
        ax.scatter(x_positions, pred_values, alpha=0.6, s=30, c='red', edgecolors='black', linewidth=0.5)
        
        # Add subject labels
        for j, (true_val, pred_val) in enumerate(zip(true_values, pred_values)):
            ax.annotate(f'S{j+1}', (1, true_val), xytext=(5, 0), 
                       textcoords='offset points', fontsize=6, alpha=0.7, color='blue')
            ax.annotate(f'S{j+1}', (2, pred_val), xytext=(5, 0), 
                       textcoords='offset points', fontsize=6, alpha=0.7, color='red')
        
        # Calculate statistics
        true_mean = np.mean(true_values)
        true_std = np.std(true_values, ddof=1)
        pred_mean = np.mean(pred_values)
        pred_std = np.std(pred_values, ddof=1)
        
        # Styling
        ax.set_ylabel(f'{metric_names[metric_key]} ({metric_units[metric_key]})', fontsize=10)
        ax.set_title(f'{metric_names[metric_key]} - Distribution Analysis', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f'True: {true_mean:.1f}±{true_std:.1f}\nPred: {pred_mean:.1f}±{pred_std:.1f}\nN={len(true_values)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set y-axis limits with some padding
        all_values = np.concatenate([true_values, pred_values])
        y_min = np.min(all_values) - 0.1 * (np.max(all_values) - np.min(all_values))
        y_max = np.max(all_values) + 0.1 * (np.max(all_values) - np.min(all_values))
        ax.set_ylim(y_min, y_max)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if config and hasattr(config, 'FIGURES_CLASSIFICATION_DIR'):
        save_path = os.path.join(config.FIGURES_CLASSIFICATION_DIR, 'distribution_analysis.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}\n")
    
    plt.close(fig)


def _perform_paired_t_tests(all_true_metrics: list, all_pred_metrics: list, metric_keys: list, metric_names: dict, metric_units: dict, config: dict) -> None:
    """
    Perform paired t-tests to compare true vs predicted sleep metrics.
    
    Args:
        all_true_metrics (list): List of true metrics for each subject
        all_pred_metrics (list): List of predicted metrics for each subject
        metric_keys (list): List of metric keys to test
        metric_names (dict): Display names for metrics
        metric_units (dict): Units for metrics
        config (module): Configuration module
    """
    print("Performing paired t-tests...")
    print("=" * 95)
    print(f"{'Metric':<25} {'True Mean±SD':<15} {'Pred Mean±SD':<15} {'t-statistic':<12} {'p-value':<10} {'Significant':<12}")
    print("=" * 95)
    
    significant_tests = 0
    total_tests = 0
    
    for metric_key in metric_keys:
        # Collect data points from all subjects
        true_values = []
        pred_values = []
        
        for true_metrics, pred_metrics in zip(all_true_metrics, all_pred_metrics):
            if (metric_key in true_metrics and metric_key in pred_metrics and 
                true_metrics[metric_key] is not None and pred_metrics[metric_key] is not None):
                true_values.append(true_metrics[metric_key])
                pred_values.append(pred_metrics[metric_key])
        
        if len(true_values) < 3:
            print(f"{metric_names[metric_key]:<25} {'Insufficient data':<15} {'Insufficient data':<15} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
            continue
        
        true_values = np.array(true_values)
        pred_values = np.array(pred_values)
        
        # Perform paired t-test
        t_stat, p_value = ttest_rel(true_values, pred_values)
        
        # Calculate means and standard deviations
        true_mean = np.mean(true_values)
        true_std = np.std(true_values, ddof=1)
        pred_mean = np.mean(pred_values)
        pred_std = np.std(pred_values, ddof=1)
        
        # Determine significance (α = 0.05)
        is_significant = p_value < 0.05
        if is_significant:
            significant_tests += 1
        total_tests += 1
        
        # Format output
        true_str = f"{true_mean:.1f}±{true_std:.1f}"
        pred_str = f"{pred_mean:.1f}±{pred_std:.1f}"
        t_str = f"{t_stat:.3f}"
        p_str = f"{p_value:.4f}"
        sig_str = "Yes" if is_significant else "No"
        
        print(f"{metric_names[metric_key]:<25} {true_str:<15} {pred_str:<15} {t_str:<12} {p_str:<10} {sig_str:<12}")
    
    print("=" * 95)
    print(f"Summary: {significant_tests}/{total_tests} metrics showed significant differences (p < 0.05)")
    print("\n" + "=" * 95)

    print("Statistical Interpretation:")
    print("- p < 0.05: Significant difference between true and predicted values")
    print("- p ≥ 0.05: No significant difference (values are statistically similar)")
    print("=" * 95)


def _plot_from_sleep_metrics(sleep_metrics_data: dict, config: dict) -> None:
    """
    Create correlation plots from pre-calculated sleep metrics.
    
    Args:
        sleep_metrics_data (dict): Pre-calculated sleep metrics
        config (module): Configuration module
    """
    all_true_metrics = sleep_metrics_data['all_true_metrics']
    all_pred_metrics = sleep_metrics_data['all_pred_metrics']
    
    # Define metrics to plot (numeric metrics only)
    metric_keys = ['SOL', 'TST', 'SE', 'WASO', 'REM_latency', 'n_awakenings', 'REM_cycles', 'REM_duration']
    metric_names = {
        'SOL': 'Sleep Onset Latency',
        'TST': 'Total Sleep Time', 
        'SE': 'Sleep Efficiency',
        'WASO': 'Wake After Sleep Onset',
        'REM_latency': 'REM Latency',
        'n_awakenings': 'Number of Awakenings',
        'REM_cycles': 'REM Cycles',
        'REM_duration': 'REM Duration'
    }
    metric_units = {
        'SOL': 'min',
        'TST': 'min',
        'SE': '%',
        'WASO': 'min', 
        'REM_latency': 'min',
        'n_awakenings': 'count',
        'REM_cycles': 'count',
        'REM_duration': 'min'
    }
    
    if all_true_metrics and all_pred_metrics:
        _create_bland_altman_plots(all_true_metrics, all_pred_metrics, 
                                        metric_keys, metric_names, metric_units, config)
        _create_correlation_plots(all_true_metrics, all_pred_metrics, 
                                metric_keys, metric_names, metric_units, config)
        _create_distribution_plots(all_true_metrics, all_pred_metrics, 
                                 metric_keys, metric_names, metric_units, config)
        _perform_paired_t_tests(all_true_metrics, all_pred_metrics, 
                              metric_keys, metric_names, metric_units, config)