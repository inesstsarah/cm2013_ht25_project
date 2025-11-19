import config
from src.data_loader import load_all_holdout_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.inference import make_inference, generate_submission_file
from src.utils import save_cache, load_cache
import os
import joblib

def run_inference():
    print(f"--- Sleep Scoring Inference - Iteration {config.CURRENT_ITERATION} ---")

    # Load the trained model (assuming it was saved during training)
    model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
    model = load_cache(model_filename, config.CACHE_DIR)
    if model is None:
        print("Error: Trained model not found. Please run main.py first to train a model.")
        return

    # 1. Load Hold-out Data
    holdout_multi_channel_data, holdout_record_ids, channel_info = load_all_holdout_data(config.HOLDOUT_DIR)

    # 2. Preprocessing (using the same logic as training)
    preprocessed_holdout_data = None
    cache_filename_preprocess_holdout = f"preprocessed_holdout_data_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        preprocessed_holdout_data = load_cache(cache_filename_preprocess_holdout, config.CACHE_DIR)
    
    if preprocessed_holdout_data is None:
        preprocessed_holdout_data = preprocess(holdout_multi_channel_data, channel_info, config)
        if config.USE_CACHE:
            save_cache(preprocessed_holdout_data, cache_filename_preprocess_holdout, config.CACHE_DIR)

    # 3. Feature Extraction (using the same logic as training)
    holdout_features = None
    cache_filename_features_holdout = f"features_holdout_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        holdout_features = load_cache(cache_filename_features_holdout, config.CACHE_DIR)

    if holdout_features is None:
        holdout_features = extract_features(preprocessed_holdout_data, channel_info, config)
        if config.USE_CACHE:
            save_cache(holdout_features, cache_filename_features_holdout, config.CACHE_DIR)

    # 4. Feature Selection
    cache_filename_selected_indices = f"selected_indices_iter{config.CURRENT_ITERATION}.joblib"
    selected_indices = load_cache(cache_filename_selected_indices, config.CACHE_DIR)
    selected_features = holdout_features[:, selected_indices]

    # 5. Make Inference
    predictions = make_inference(model, selected_features, config)

    epoch_numbers = list(range(len(predictions)))

    # 5. Generate Submission File
    generate_submission_file(predictions, holdout_record_ids, epoch_numbers, config)

    print("--- Inference Finished ---")

if __name__ == "__main__":
    run_inference()
