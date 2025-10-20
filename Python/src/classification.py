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


# TODO: Statistical comparison between iterations (t-test on kappa scores)
# Clinical plausibility check
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

