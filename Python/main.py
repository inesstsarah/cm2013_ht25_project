import config
from src.data_loader import load_all_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features, get_feature_names
from src.feature_selection import select_features
from src.classification import train_classifier
from src.visualization import visualize_results
from src.report import generate_report
from src.utils import save_cache, load_cache
import sys
import io
import time
from datetime import timedelta


class TeeIO:
    """ 
        A class to duplicate stdout to both terminal and a string buffer.
    """
    def __init__(self, original_stdout, string_buffer, show_terminal=True):
        self.original_stdout = original_stdout
        self.string_buffer = string_buffer
        self.show_terminal = show_terminal

    def write(self, text):
        if self.show_terminal:
            self.original_stdout.write(text)
        self.string_buffer.write(text)

    def flush(self):
        if self.show_terminal:
            self.original_stdout.flush()
        self.string_buffer.flush()

def main():
    # Create a string buffer
    stdout_buffer = io.StringIO()
    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to the buffer
    # sys.stdout = stdout_buffer

    # Redirect stdout to both terminal and buffer
    sys.stdout = TeeIO(original_stdout, stdout_buffer, show_terminal=True)

    print("=== PROCESSING LOG ===")

    print(f"--- Sleep Scoring Pipeline - Iteration {config.CURRENT_ITERATION} ---")
    
    # Start total timing
    total_start_time = time.time()
    step_times = {}

    # 1. Load Data
    # Example uses R1.edf and R1.xml - students should adapt for their dataset
    print("\n=== STEP 1: DATA LOADING ===")
    step_start = time.time()
    # Handle both new multi-channel format and old single-channel format for compatibility
    try:
        multi_channel_data, labels, record_ids, channel_info = load_all_training_data(config.TRAINING_DIR)
        print(f"Multi-channel data loaded:")
        print(f"  EEG: {multi_channel_data['eeg'].shape}")
        print(f"  EOG: {multi_channel_data['eog'].shape}")
        print(f"  EMG: {multi_channel_data['emg'].shape}")
        print(f"Labels shape: {labels.shape}")

    except (ValueError, TypeError):
        print("Fail to load multi-channel data, closely check the error message above.")
    step_times['Data Loading'] = time.time() - step_start

    # 2. Preprocessing
    print("\n=== STEP 2: PREPROCESSING ===")
    step_start = time.time()
    preprocessed_data = None
    cache_filename_preprocess = f"preprocessed_data_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
        if preprocessed_data is not None:
            print("Loaded preprocessed data from cache")

    if preprocessed_data is None:
        preprocessed_data = preprocess(multi_channel_data, channel_info, config)
        print(f"Preprocessed EEG shape: {preprocessed_data['eeg'].shape}")
        if config.USE_CACHE:
            save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
            print("Saved preprocessed data to cache")
    step_times['Preprocessing'] = time.time() - step_start

    # 3. Feature Extraction
    print("\n=== STEP 3: FEATURE EXTRACTION ===")
    step_start = time.time()
    features = None
    cache_filename_features = f"features_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        features = load_cache(cache_filename_features, config.CACHE_DIR)
        if features is not None:
            print("Loaded features from cache")

    if features is None:
        features = extract_features(preprocessed_data, channel_info, config)
        print(f"Extracted features shape: {features.shape}")
        if features.shape[1] == 0:
            print("⚠️  WARNING: No features extracted! Students must implement feature extraction.")
        if config.USE_CACHE:
            save_cache(features, cache_filename_features, config.CACHE_DIR)
            print("Saved features to cache")

    feature_names = get_feature_names(preprocessed_data, channel_info, config)
    step_times['Feature Extraction'] = time.time() - step_start
    # 4. Feature Selection
    print("\n=== STEP 4: FEATURE SELECTION ===")
    step_start = time.time()
    selected_features, selected_indices = select_features(features, labels, config)
    print(f"Selected features shape: {selected_features.shape}")

    cache_filename_selected_indices = f"selected_indices_iter{config.CURRENT_ITERATION}.joblib"
    save_cache(selected_indices, cache_filename_selected_indices, config.CACHE_DIR)
    print("Saved selected indices to cache")
    selected_feature_names = [feature_names[i] for i in selected_indices]
    for i, name in enumerate(selected_feature_names):
        if i % 4 == 0:
            print()  
            print("  ", end="") 
        print(f"{name:30s}", end="")
    print()
    step_times['Feature Selection'] = time.time() - step_start

    # 5. Classification
    print("\n=== STEP 5: CLASSIFICATION ===")
    step_start = time.time()
    cache_filename_model = f"model_iter{config.CURRENT_ITERATION}.joblib"
    cache_filename_scaler = f"scaler_iter{config.CURRENT_ITERATION}.joblib"
    if selected_features.shape[1] > 0:
        model = train_classifier(selected_features, labels, record_ids, config)
        print(f"Trained {config.CLASSIFIER_TYPE} classifier")
        if config.USE_CACHE:
            save_cache(model['model'], cache_filename_model, config.CACHE_DIR)
            save_cache(model['scaler'], cache_filename_scaler, config.CACHE_DIR)
            print("Saved model and scaler to cache")
    else:
        print("⚠️  WARNING: Cannot train classifier - no features available!")
        print("Students must implement feature extraction first.")
        model = None
    step_times['Classification'] = time.time() - step_start

    # 6. Visualization
    print("\n=== STEP 6: VISUALIZATION ===")
    step_start = time.time()
    if model is not None:
        visualize_results(model, record_ids, config)
    else:
        print("Skipping visualization - no trained model")
    step_times['Visualization'] = time.time() - step_start

    # 7. Report Generation
    print("\n=== STEP 7: PROCESSING LOG & REPORT GENERATION ===")
    # Calculate total time
    total_time = time.time() - total_start_time
    step_times['Total'] = total_time

    # Print timing summary
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    for step_name, step_time in step_times.items():
        if step_name == 'Total':
            print(f"{step_name:25s}: {step_time:8.2f} seconds ({str(timedelta(seconds=int(step_time)))})")
        else:
            percentage = (step_time / total_time) * 100
            print(f"{step_name:25s}: {step_time:8.2f} seconds ({percentage:5.1f}%)")
    print("="*60)
    
    # Restore the original stdout
    sys.stdout = original_stdout

    # Get the captured output from the buffer
    processing_log = stdout_buffer.getvalue()   
     
    if model is not None:
        generate_report(model, selected_features, labels, config, processing_log)
    else:
        print("Skipping report - no trained model")


if __name__ == "__main__":
    main()
    # from src.visualization import plot_sample_epoch
    # plot_sample_epoch('../data/Holdout/H1.edf', epoch_idx=10)
    # plot_sample_epoch('../data/training/R1.edf', epoch_idx=0)
