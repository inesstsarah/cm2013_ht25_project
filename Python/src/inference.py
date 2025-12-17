import numpy as np
import pandas as pd
import os
from scipy.ndimage import median_filter

def temporal_smoothing(predictions: np.ndarray, record_ids: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply temporal smoothing to predictions to reduce unrealistic stage transitions.
    Sleep stages should be continuous in time, so we smooth predictions within each record.
    
    Args:
        predictions (np.ndarray): Raw predictions
        record_ids (np.ndarray): Record IDs for each prediction
        window_size (int): Size of the smoothing window (must be odd)
    
    Returns:
        np.ndarray: Smoothed predictions
    
    Example:
        >>> smoothed_preds = temporal_smoothing(predictions, record_ids, window_size=3)
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    smoothed_predictions = predictions.copy()
    unique_records = np.unique(record_ids)
    
    for record_id in unique_records:
        mask = record_ids == record_id
        record_predictions = predictions[mask]
        
        # Apply median filter for temporal smoothing
        # This helps remove isolated misclassifications
        smoothed_record = median_filter(record_predictions, size=window_size, mode='nearest')
        smoothed_predictions[mask] = smoothed_record
    
    return smoothed_predictions

def make_inference(model, holdout_data, config, record_ids=None, apply_smoothing=True):
    """
    Makes predictions on the hold-out data using the trained model.

    Args:
        model (object): The trained classification model.
        holdout_data (np.ndarray): The preprocessed and feature-extracted hold-out data.
        config (module): The configuration module.
        record_ids (np.ndarray, optional): Record IDs for temporal smoothing.
        apply_smoothing (bool): Whether to apply temporal smoothing (default: True).

    Returns:
        np.ndarray: Predicted labels for the hold-out data.
    
    Example:
        >>> predictions = make_inference(model, holdout_data, config, record_ids, apply_smoothing=True)
    """
    print("Making inference on hold-out data...")
    predictions = model.predict(holdout_data)
    
    # Apply temporal smoothing if record_ids are provided
    if apply_smoothing and record_ids is not None:
        print("Applying temporal smoothing to predictions...")
        predictions = temporal_smoothing(predictions, record_ids, window_size=3)
        print("Temporal smoothing completed.")
    
    return predictions

def generate_submission_file(predictions, record_numbers, epoch_numbers, config):
    """
    Generates a submission CSV file.

    Args:
        predictions (np.ndarray): The predicted sleep stage labels.
        record_numbers (list): List of record numbers corresponding to each epoch.
        epoch_numbers (list): List of epoch numbers corresponding to each epoch.
        config (module): The configuration module.
        
    Returns:
        None

    Example:
        >>> generate_submission_file(predictions, record_numbers, epoch_numbers, config)
    """
    print(f"Generating submission file: {config.SUBMISSION_FILE}...")
    submission_df = pd.DataFrame({
        'record_number': record_numbers,
        'epoch_number': epoch_numbers,
        'label': predictions
    })
    submission_df.to_csv(os.path.join(config.DATA_DIR, config.SUBMISSION_FILE), index=False)
    print("Submission file generated successfully.")
