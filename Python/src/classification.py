import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

def hyperparameter_optimization(X_train, y_train):
    """Function to search for the optimal parameters in a hyperparameter space"""
    
    grid_params = { 'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
    
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=3, n_jobs = -1)
    # fit the model on our train set
    g_res = gs.fit(X_train, y_train)
    # find the best score
    best_score = g_res.best_score_

    # Get dictionary of best params
    best_params = g_res.best_params_
    return(best_score, best_params)



    
    

def train_classifier(features, labels, config):
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

    # BASIC train/test split - students should implement cross-validation
    # TODO: Students should implement k-fold cross-validation for more robust evaluation
    # Use stratified split for realistic sleep data distribution
    # Sleep stages are naturally imbalanced (more N2, less N1/REM)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print("Using stratified train/test split to maintain class balance")
    except ValueError as e:
        # Fallback for edge cases (very small datasets)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        print(f"Using non-stratified split: {e}")
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # TODO: Students should address class imbalance in sleep data:
    # - Sleep stages are not equally distributed
    # - Consider SMOTE, class weights, or other techniques
    # from imblearn.over_sampling import SMOTE
    # smote = SMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

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
    if (config.USE_HYPERPARAMETER_OPT == False):
        model.fit(X_train, y_train)

    else: # Do hyperparameter optimization
        # Hyperparameter optimization 
        _, best_hyperparameters = hyperparameter_optimization(X_train, y_train)
        # Unpack results of best_hyperparameters
        model = KNeighborsClassifier(**best_hyperparameters)
        model.fit(X_train, y_train)

    # Comprehensive evaluation with detailed performance metrics
    y_pred = model.predict(X_test)
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {overall_accuracy:.3f}")

    # Getting probabilities for AUC-ROC score
    y_probs = model.predict_proba(X_test)

    # Calculate and display detailed performance metrics
    print_performance_metrics(y_test, y_pred)

    # TODO: Students should add more advanced metrics:

    # - ROC-AUC for each class
    roc_auc = roc_auc_score(y_test, y_probs)

    # NOTE: not sure whether to print it here or if we have to print it in the print_performance_metrics function
    print(f"ROC-AUC score: {roc_auc}")
    # Generate plot for ROC-AUC

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr)



    # - Cross-validation scores
    # - Feature importance analysis
    print("\nTODO: Students should add Cohen's kappa and ROC-AUC metrics")

    return model


def print_performance_metrics(y_true, y_pred):
    """
    Print comprehensive performance metrics for sleep stage classification.

    Includes accuracy, sensitivity (recall), specificity, and F1-score for each sleep stage.
    """

    # Sleep stage labels and names (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    stage_labels = list(range(5))

    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    # Specificity - True Negative Rate
    specificity = compute_specificity(y_true, y_pred, stage_labels)

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
    auc_results = compute_auc(y_true, y_pred_proba)

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
    print_confusion_matrix(y_true, y_pred, stage_names, stage_labels)
    
    # Class distribution in test set
    print_sleep_stage_distribution(y_true)

    # Sleep scoring specific notes
    print_scoring_notes()

    # Clinical plausibility check
    compare_sleep_metrics(y_true, y_pred,record_ids)

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


def compute_auc(y_true, y_pred_proba):
    """
    Compute per-class and macro ROC-AUC.

    Args:
        y_true (array): true labels
        y_pred_proba (array): predicted probabilities for each class

    Returns:
        dict: containing per-class and macro ROC-AUC scores
    """
    # Translate to one-hot matrix
    n_classes = y_pred_proba.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]

    #  Compute ROC-AUC
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
    
    return {
        'auc_per_class': auc_per_class,
        'macro_auc': macro_auc,
        'weighted_auc': weighted_auc
    }


def compute_specificity(y_true, y_pred, stage_label):
    specificity = []
    for i in range(len(stage_label)):
        tn = np.sum((y_true != i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(spec)
    return specificity


def print_confusion_matrix(y_true, y_pred, stage_names, stage_labels):
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=stage_labels) 
    # Create a formatted confusion matrix
    cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
    print(cm_df.to_string())


def print_sleep_stage_distribution(y_true):
    print("\nClass Distribution in Test Set:")
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    unique, counts = np.unique(y_true, return_counts=True)
    total_samples = len(y_true)

    for stage_idx, count in zip(unique, counts):
        stage_name = stage_names[stage_idx]
        percentage = count / total_samples * 100
        print(f"{stage_name}: {count} samples ({percentage:.1f}%)")


def print_scoring_notes():
    print("\nNotes for Sleep Scoring:")
    print("- Sensitivity = Recall = True Positive Rate (correctly identified stages)")
    print("- Specificity = True Negative Rate (correctly rejected stages)")
    print("- Sleep stage imbalance is natural (more N2, less N1/REM)")
    print("- Consider Cohen's kappa for chance-corrected agreement")
    print("- Clinical focus: High sensitivity for REM and N3 stages")
