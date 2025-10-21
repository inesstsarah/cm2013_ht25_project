import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import pandas as pd
from imblearn.over_sampling import SMOTE


def train_classifier(features, labels, record_ids, config):
    """
    STUDENT IMPLEMENTATION AREA: Train classifier based on iteration.

    This function provides a basic framework but students should enhance it:

    1. Implement proper cross-validation (not just train/test split)
    2. Address class imbalance in sleep stage data
    3. Tune hyperparameters for each classifier
    4. Add more sophisticated evaluation metrics
    5. Consider ensemble methods in later iterations

    Args:
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.

    Returns:
        object: The trained classifier.
    """
    print(f"Training {config.CLASSIFIER_TYPE} classifier...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # Basic validation
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("No features available for training!")

    # Select classifier based on iteration (using config parameters)
    if config.CURRENT_ITERATION == 1:
        # Iteration 1: Simple k-NN
        model = KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)
        print(f"Using k-NN with k={config.KNN_N_NEIGHBORS}")
    elif config.CURRENT_ITERATION == 2:
        # Iteration 2: SVM
        # TODO: Students should tune hyperparameters (C, kernel, gamma)
        model = SVC(
            C=getattr(config, 'SVM_C', 1.0),
            kernel=getattr(config, 'SVM_KERNEL', 'rbf'),
            random_state=42
        )
        print(f"Using SVM with C={model.C}, kernel={model.kernel}")
    elif config.CURRENT_ITERATION >= 3:
        # Iteration 3+: Random Forest
        # TODO: Students should tune hyperparameters (n_estimators, max_depth, etc.)
        model = RandomForestClassifier(
            n_estimators=getattr(config, 'RF_N_ESTIMATORS', 100),
            max_depth=getattr(config, 'RF_MAX_DEPTH', None),
            min_samples_split=getattr(config, 'RF_MIN_SAMPLES_SPLIT', 2),
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        print(f"Using Random Forest with {model.n_estimators} trees")
    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    # Train the model
    print("Training model...")
    _LOGO_split_training(model, features, labels, record_ids)

    return model


def compare_sleep_metrics(y_true, y_pred, record_ids=None, epoch_duration=30):
    """
    Compare sleep architecture metrics between ground truth and predictions.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels  
        record_ids (np.ndarray): Optional record identifiers
        epoch_duration (int): Duration of each epoch in seconds
        
    Returns:
        dict: Comparison results for each record
    """
    if record_ids is None:
        # Overall comparison
        true_metrics = _calculate_sleep_metrics(y_true, epoch_duration)
        pred_metrics = _calculate_sleep_metrics(y_pred, epoch_duration)
        
        print("\nSleep Architecture Metrics Comparison (Overall)")
        print("=" * 80)
        _print_metrics_comparison_table(true_metrics, pred_metrics)
        
        return {
            'overall': {
                'true_metrics': true_metrics,
                'pred_metrics': pred_metrics
            }
        }
    else:
        # Per-record comparison
        unique_records = np.unique(record_ids)
        results = {}
        
        for record_id in unique_records:
            indices = np.where(record_ids == record_id)[0]
            if len(indices) == 0:
                continue
                
            true_labels = y_true[indices]
            pred_labels = y_pred[indices]
            
            true_metrics = _calculate_sleep_metrics(true_labels, epoch_duration)
            pred_metrics = _calculate_sleep_metrics(pred_labels, epoch_duration)
            
            print(f"\nSleep Architecture Metrics Comparison - {record_id}")
            print("=" * 80)
            _print_metrics_comparison_table(true_metrics, pred_metrics)
            
            results[record_id] = {
                'true_metrics': true_metrics,
                'pred_metrics': pred_metrics
            }
            
        return results


# TODO: Statistical comparison between iterations (t-test on kappa scores)
def _LOGO_split_training(model, features, labels, record_ids):
    # Create LOSO cross-validation split
    logo = LeaveOneGroupOut()
    loso_results = []
    smote = SMOTE(random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(features, labels, groups=record_ids)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # - Sleep stages are not equally distributed
        # Sleep stages are naturally imbalanced (more N2, less N1/REM)
        # TODO：Use class weighting method in next iterations
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Which subject is held out in this fold?
        train_subjects = np.unique(record_ids[train_idx])
        test_subject = np.unique(record_ids[test_idx])[0]
        print(f"\nFold {fold_idx+1}/{len(train_subjects)+1}: Training on {len(train_subjects)} subjects, testing on {test_subject}")

        # Train classifier on 9 subjects
        model.fit(X_train, y_train)

        # Predict on held-out subject
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        eva_results = _training_evaluation(y_test, y_pred, y_pred_proba)

        loso_results.append({
            'subject': test_subject,
            **eva_results
            })

    # Report mean ± std across all subjects
    mean_acc = np.mean([r['accuracy'] for r in loso_results])
    std_acc = np.std([r['accuracy'] for r in loso_results])
    mean_kappa = np.mean([r['kappa'] for r in loso_results])
    std_kappa = np.std([r['kappa'] for r in loso_results])

    print("\n" + "="*60)
    print(f"LOSO Cross-Validation Results ({len(loso_results)} subjects):")
    print(f"  Accuracy = {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"  Kappa    = {mean_kappa:.3f} ± {std_kappa:.3f}")
    print("="*60)


def _training_evaluation(y_true, y_pred, y_pred_proba):
    # Calculate metrics for this subject
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_labels = list(range(5))

    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    # Specificity - True Negative Rate
    specificity = _compute_specificity(y_true, y_pred, stage_labels)

    # Macro and weighted averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_specificity = np.mean(specificity)
    macro_f1 = np.mean(f1)

    weighted_precision = np.sum(precision * support) / np.sum(support)
    weighted_recall = np.sum(recall * support) / np.sum(support)
    weighted_specificity = np.sum(np.array(specificity) * support) / np.sum(support)
    weighted_f1 = np.sum(f1 * support) / np.sum(support)

    # AUC
    auc_results = _compute_auc(y_true, y_pred_proba)

    # Detailed report
    print("Per-Class Performance Metrics:")
    print("-" * 80)
    print(f"{'Stage':<15} {'Precision':<10} {'Recall':<10} {'Specificity':<12} {'F1-Score':<10} {'ROC-AUC':<10} {'Support':<8}")
    print("-" * 80)
    for i, stage_name in enumerate(stage_names):
        print(f"{stage_name:<15} {precision[i]:<10.3f} {recall[i]:<10.3f} {specificity[i]:<12.3f} {f1[i]:<10.3f} {auc_results['auc_per_class'][i]:<10.3f} {support[i]:<8}")
    print("-" * 80)

    print(f"{'Macro':<15} {macro_precision:<10.3f} {macro_recall:<10.3f} {macro_specificity:<12.3f} {macro_f1:<10.3f} {auc_results['macro_auc']:<10.3f} {np.sum(support):<8}")
    print(f"{'Weighted':<15} {weighted_precision:<10.3f} {weighted_recall:<10.3f} {weighted_specificity:<12.3f} {weighted_f1:<10.3f} {auc_results['weighted_auc']:<10.3f} {np.sum(support):<8}")
    print(f"{'Accuracy':<15} {accuracy:<10.3f} {'-':<10} {'-':<12} {'-':<10} {'-':<10} {len(y_true):<8}")
    print(f"{'Cohen Kappa':<15} {kappa:<10.3f} {'-':<10} {'-':<12} {'-':<10} {'-':<10} {np.sum(support):<8}")
    print("-" * 80)

    # Confusion Matrix
    _print_confusion_matrix(y_true, y_pred, stage_names, stage_labels)
    
    # Class distribution in test set
    _print_sleep_stage_distribution(y_true)

    # Sleep scoring specific notes
    _print_scoring_notes()

    # Sleep architecture metrics
    compare_sleep_metrics(y_true, y_pred)

    result = {
            'accuracy': accuracy,
            'kappa': kappa,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1,
            'specificity': specificity
    }
    return result


def _compute_auc(y_true, y_pred_proba):
    """
    Compute per-class and macro ROC-AUC.

    Args:
        y_true (array): true labels
        y_pred_proba (array): predicted probabilities for each class

    Returns:
        dict: containing per-class and macro ROC-AUC scores
    """
    # === Step 1: Translate to one-hot matrix ===
    n_classes = y_pred_proba.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]

    # === Step 2: Compute ROC-AUC ===
    auc_per_class = []
    for i in range(n_classes):
        try:
            auc = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
        except ValueError:
            auc = np.nan  # if class missing
        auc_per_class.append(auc)

    macro_auc = np.nanmean(auc_per_class)

    unique, counts = np.unique(y_true, return_counts=True)
    weights = np.zeros(n_classes)
    weights[unique] = counts / np.sum(counts)

    weighted_auc = np.nansum(auc_per_class * weights)

    # === Step 3: Return ===
    return {
        'auc_per_class': auc_per_class,
        'macro_auc': macro_auc,
        'weighted_auc': weighted_auc
    }


def _compute_specificity(y_true, y_pred, stage_label):
    specificity = []
    for i in range(len(stage_label)):
        tn = np.sum((y_true != i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(spec)
    return specificity


def _print_confusion_matrix(y_true, y_pred, stage_names, stage_labels):
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=stage_labels)

    # Create a formatted confusion matrix
    cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
    print(cm_df.to_string())


def _print_sleep_stage_distribution(y_true):
    print("\nClass Distribution in Test Set:")
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    unique, counts = np.unique(y_true, return_counts=True)
    total_samples = len(y_true)

    for stage_idx, count in zip(unique, counts):
        stage_name = stage_names[stage_idx]
        percentage = count / total_samples * 100
        print(f"{stage_name}: {count} samples ({percentage:.1f}%)")


def _print_scoring_notes():
    print("\nNotes for Sleep Scoring:")
    print("- Sensitivity = Recall = True Positive Rate (correctly identified stages)")
    print("- Specificity = True Negative Rate (correctly rejected stages)")
    print("- Sleep stage imbalance is natural (more N2, less N1/REM)")
    print("- Consider Cohen's kappa for chance-corrected agreement")
    print("- Clinical focus: High sensitivity for REM and N3 stages")


# Clinical plausibility check
def _calculate_sleep_metrics(labels, epoch_duration=30):
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


def _print_metrics_comparison_table(true_metrics, pred_metrics):
    """
    Print formatted comparison table for sleep metrics.
    
    Args:
        true_metrics (dict): Ground truth sleep metrics
        pred_metrics (dict): Predicted sleep metrics
    """
    # Define metric display names and units
    metric_info = {
        'SOL': ('Sleep Onset Latency', 'min'),
        'TST': ('Total Sleep Time', 'min'),
        'SE': ('Sleep Efficiency', '%'),
        'WASO': ('Wake After Sleep Onset', 'min'),
        'REM_latency': ('REM Latency', 'min'),
        'n_awakenings': ('Number of Awakenings', 'count'),
        'REM_cycles': ('REM Cycles', 'count'),
        'REM_duration': ('REM Duration', 'min')
    }
    
    # Print basic metrics
    print(f"{'Metric':<25} {'Ground Truth':<15} {'Predicted':<15} {'Error':<10} {'Unit':<8}")
    print("-" * 80)
    
    for metric_key, (display_name, unit) in metric_info.items():
        if metric_key in true_metrics and metric_key in pred_metrics:
            true_val = true_metrics[metric_key]
            pred_val = pred_metrics[metric_key]
            
            # Handle None values
            if true_val is None or pred_val is None:
                true_str = "N/A" if true_val is None else f"{true_val:.1f}"
                pred_str = "N/A" if pred_val is None else f"{pred_val:.1f}"
                error_str = "N/A"
            else:
                true_str = f"{true_val:.1f}"
                pred_str = f"{pred_val:.1f}"
                error = abs(pred_val - true_val)
                error_str = f"{error:.1f}"
            
            print(f"{display_name:<25} {true_str:<15} {pred_str:<15} {error_str:<10} {unit:<8}")
    
    # Print sleep stage percentages
    print("\nSleep Stage Percentages (relative to TST):")
    print("-" * 60)
    print(f"{'Stage':<10} {'Ground Truth':<15} {'Predicted':<15} {'Error':<10}")
    print("-" * 60)
    
    if 'sleep_stage_percentages' in true_metrics and 'sleep_stage_percentages' in pred_metrics:
        true_stages = true_metrics['sleep_stage_percentages']
        pred_stages = pred_metrics['sleep_stage_percentages']
        
        for stage in ['N1', 'N2', 'N3', 'REM']:
            if stage in true_stages and stage in pred_stages:
                true_pct = true_stages[stage]
                pred_pct = pred_stages[stage]
                error = abs(pred_pct - true_pct)
                
                print(f"{stage:<10} {true_pct:<15.1f} {pred_pct:<15.1f} {error:<10.1f}")
    
    # Calculate and print summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)
    
    # Calculate mean absolute error for numeric metrics
    numeric_metrics = ['SOL', 'TST', 'SE', 'WASO', 'REM_latency', 'n_awakenings', 'REM_cycles', 'REM_duration']
    mae_values = []
    
    for metric in numeric_metrics:
        if (metric in true_metrics and metric in pred_metrics and 
            true_metrics[metric] is not None and pred_metrics[metric] is not None):
            error = abs(pred_metrics[metric] - true_metrics[metric])
            mae_values.append(error)
    
    if mae_values:
        mean_mae = np.mean(mae_values)
        print(f"Mean Absolute Error (numeric metrics): {mean_mae:.2f}")
    
    # Calculate MAE for stage percentages
    if ('sleep_stage_percentages' in true_metrics and 
        'sleep_stage_percentages' in pred_metrics):
        stage_errors = []
        for stage in ['N1', 'N2', 'N3', 'REM']:
            if stage in true_metrics['sleep_stage_percentages'] and stage in pred_metrics['sleep_stage_percentages']:
                error = abs(pred_metrics['sleep_stage_percentages'][stage] - 
                           true_metrics['sleep_stage_percentages'][stage])
                stage_errors.append(error)
        
        if stage_errors:
            mean_stage_mae = np.mean(stage_errors)
            print(f"Mean Absolute Error (stage percentages): {mean_stage_mae:.2f}%")
    
    print("=" * 80)

