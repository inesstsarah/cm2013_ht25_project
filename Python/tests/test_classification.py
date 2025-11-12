"""
Tests for the classification module
This can be used to test the functions in the classification module 
"""
import pytest
import os
import sys
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import config



def get_model_from_config(config):
    """Get model instance based on config CLASSIFIER_TYPE"""
    if config.CLASSIFIER_TYPE == 'knn':
        return KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)
    
    else:
        raise ValueError(f"Unknown classifier type: {config.CLASSIFIER_TYPE}")

def test_hyperparameter_optimization(X_train, y_train, config):
    """
    Function to search for the optimal parameters in a hyperparameter space

    Args:
        X_train (np.ndarray[float]): Array of training set variables
        y_train (np.ndarray[int]): Array of training set classes
        config (dict): Config for repository.
    """
    
    # Get model and parameters from config
    base_model = get_model_from_config(config)
    grid_params = config.GRID_PARAMS

    # Perform grid search
    gs = GridSearchCV(base_model, grid_params, verbose=1, cv=3, n_jobs=-1)
    g_res = gs.fit(X_train, y_train)

    # find the best score
    best_score = g_res.best_score_

    # Get dictionary of best params
    best_params = g_res.best_params_
    return(best_score, best_params)
