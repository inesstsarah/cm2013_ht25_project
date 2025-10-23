"""
Tests for the classification module
This can be used to test the functions in the classification module 
"""
import pytest
import os
import sys
from src.classification import hyperparameter_optimization
from sklearn import datasets

def test_hyperparameter_optimization():
    # TODO: Add to this test by using artificial data and using the hyperparameter optimization function
    X_train, y_train = datasets.make_classification(
    n_samples=100)
    '''X_train = [[4, 5, 10, 4, 3, 11, 14 , 8, 10, 12]]
    y_train = [21, 21, 24, 24, 21, 24, 24, 21, 21, 21]'''
    best_score, best_params = hyperparameter_optimization(X_train, y_train)
    print(f"Best score: {best_score}")
    print(f"Best params: {best_params}")
    