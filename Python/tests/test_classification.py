"""
Tests for the classification module
This can be used to test the functions in the classification module 
"""
import pytest
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.classification import (
    _get_model_from_config,
    hyperparameter_optimization,
    _LOSO_split_training,
    _training_evaluation,
    _compute_auc,
    _compute_specificity,
    _compare_sleep_metrics
)


def test_get_model_from_config():
    # Test KNN
    knn_model = _get_model_from_config('knn')
    assert knn_model is not None
    assert hasattr(knn_model, 'fit')
    assert hasattr(knn_model, 'predict')
    
    # Test SVM
    svm_model = _get_model_from_config('svm')
    assert svm_model is not None
    assert hasattr(svm_model, 'fit')
    assert hasattr(svm_model, 'predict')
    assert svm_model.probability == True
    assert svm_model.random_state == 42
    
    # Test Random Forest
    rf_model = _get_model_from_config('random_forest')
    assert rf_model is not None
    assert hasattr(rf_model, 'fit')
    assert hasattr(rf_model, 'predict')
    assert rf_model.random_state == 42
    assert rf_model.n_jobs == -1
    
    # Test invalid classifier type
    with pytest.raises(ValueError):
        _get_model_from_config('invalid_type')


def test_hyperparameter_optimization():
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    
    # Create base model
    from sklearn.neighbors import KNeighborsClassifier
    base_model = KNeighborsClassifier()
    
    # Define grid parameters
    grid_params = {
        'n_neighbors': [3, 5],
        'weights': ['uniform']
    }
    
    # Run hyperparameter optimization
    best_score, best_params = hyperparameter_optimization(base_model, X_train, y_train, grid_params)
    
    # Check return values
    assert isinstance(best_score, float)
    assert isinstance(best_params, dict)
    assert 'n_neighbors' in best_params
    assert 'weights' in best_params
    assert best_params['n_neighbors'] in [3, 5]
    assert best_params['weights'] == 'uniform'


def test_compute_auc():
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    # Create true labels
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Create predicted probabilities
    y_pred_proba = np.random.rand(n_samples, n_classes)
    # Normalize to make it proper probability distribution
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Compute AUC
    auc_results = _compute_auc(y_true, y_pred_proba)
    
    # Check return structure
    assert isinstance(auc_results, dict)
    assert 'auc_per_class' in auc_results
    assert 'macro_auc' in auc_results
    assert 'weighted_auc' in auc_results
    
    # Check values
    assert len(auc_results['auc_per_class']) == n_classes
    assert isinstance(auc_results['macro_auc'], (float, np.floating))
    assert isinstance(auc_results['weighted_auc'], (float, np.floating))
    
    # Test with missing classes (should handle NaN without warnings)
    # Function now checks for sufficient classes before calling roc_auc_score
    y_true_single_class = np.zeros(n_samples, dtype=int)
    auc_results_single = _compute_auc(y_true_single_class, y_pred_proba)
    assert len(auc_results_single['auc_per_class']) == n_classes
    # All AUC values should be NaN for single class case
    assert all(np.isnan(auc) for auc in auc_results_single['auc_per_class'])
    assert np.isnan(auc_results_single['macro_auc'])


def test_compute_specificity():
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    stage_labels = list(range(n_classes))
    
    # Compute specificity
    specificity = _compute_specificity(y_true, y_pred, stage_labels)
    
    # Check return value
    assert isinstance(specificity, list)
    assert len(specificity) == n_classes
    
    # Check that all values are between 0 and 1
    for spec in specificity:
        assert 0.0 <= spec <= 1.0
    
    # Test with perfect predictions
    y_true_perfect = np.array([0, 1, 2, 3, 4] * 20)
    y_pred_perfect = y_true_perfect.copy()
    specificity_perfect = _compute_specificity(y_true_perfect, y_pred_perfect, stage_labels)
    # All specificities should be 1.0 for perfect predictions
    for spec in specificity_perfect:
        assert spec == 1.0


def test_training_evaluation():
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    record_ids = np.ones(n_samples)  # Single record
    
    # Create predicted probabilities
    y_pred_proba = np.random.rand(n_samples, n_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Run evaluation
    results = _training_evaluation(y_true, y_pred, y_pred_proba, record_ids)
    
    # Check return structure
    assert isinstance(results, dict)
    assert 'accuracy' in results
    assert 'kappa' in results
    assert 'macro_f1' in results
    assert 'weighted_f1' in results
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1_per_class' in results
    assert 'specificity' in results
    
    # Check value ranges
    assert 0.0 <= results['accuracy'] <= 1.0
    assert isinstance(results['kappa'], (float, np.floating))
    assert 0.0 <= results['macro_f1'] <= 1.0
    assert 0.0 <= results['weighted_f1'] <= 1.0
    assert len(results['precision']) == n_classes
    assert len(results['recall']) == n_classes
    assert len(results['f1_per_class']) == n_classes
    assert len(results['specificity']) == n_classes


def test_compare_sleep_metrics():
    # Create dummy sleep stage labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    np.random.seed(42)
    n_samples = 240  # 8 hours of 30-second epochs
    
    # Create realistic sleep pattern (must be integer type for calculate_sleep_metrics)
    y_true = np.zeros(n_samples, dtype=int)
    # First 20 epochs: Wake
    y_true[0:20] = 0
    # Then sleep stages
    y_true[20:60] = 1  # N1
    y_true[60:120] = 2  # N2
    y_true[120:140] = 3  # N3
    y_true[140:160] = 4  # REM
    y_true[160:200] = 2  # N2
    y_true[200:220] = 4  # REM
    y_true[220:] = 2  # N2
    
    # Create predictions (slightly different)
    y_pred = y_true.copy().astype(int)
    y_pred[50:55] = 1  # Some misclassifications
    y_pred[150:155] = 2
    
    # Test without record_ids (overall comparison)
    results_overall = _compare_sleep_metrics(y_true, y_pred, record_ids=None, epoch_duration=30)
    
    assert isinstance(results_overall, dict)
    assert 'overall' in results_overall
    assert 'true_metrics' in results_overall['overall']
    assert 'pred_metrics' in results_overall['overall']
    
    # Test with record_ids (per-record comparison)
    record_ids = np.array(['R1'] * 120 + ['R2'] * 120)
    results_per_record = _compare_sleep_metrics(y_true, y_pred, record_ids=record_ids, epoch_duration=30)
    
    assert isinstance(results_per_record, dict)
    assert 'R1' in results_per_record or len(results_per_record) > 0
    for record_id, record_results in results_per_record.items():
        assert 'true_metrics' in record_results
        assert 'pred_metrics' in record_results


def test_loso_split_training():
    """Test _LOSO_split_training function with minimal data"""
    # Create dummy data for LOSO
    np.random.seed(42)
    n_subjects = 3
    n_epochs_per_subject = 50
    n_features = 10
    
    # Create features and labels
    features = np.random.rand(n_subjects * n_epochs_per_subject, n_features)
    labels = np.random.randint(0, 5, n_subjects * n_epochs_per_subject)
    
    # Create record IDs (one per subject)
    record_ids = np.array([f'R{i+1}' for i in range(n_subjects) for _ in range(n_epochs_per_subject)])
    
    # Run LOSO training
    results = _LOSO_split_training(features, labels, record_ids, config)
    
    # Check return structure
    assert isinstance(results, dict)
    assert 'model' in results
    assert 'y_true_aggregate' in results
    assert 'y_pred_aggregate' in results
    
    # Check shapes
    assert len(results['y_true_aggregate']) == len(labels)
    assert len(results['y_pred_aggregate']) == len(labels)
    assert hasattr(results['model'], 'predict')


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
