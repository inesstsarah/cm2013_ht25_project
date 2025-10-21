"""
Tests for the classification module
This can be used to test the functions in the classification module 
"""
import pytest
import os
import sys
from src.classification import hyperparameter_optimization

def test_hyperparameter_optimization():
    # TODO: Add to this test by using artificial data and using the hyperparameter optimization function
    X_train = [[0,1,2,3,4],[9,1,2,3,4],[9,9,8,5,6],[9,9,8,5,6],[9,9,8,5,6]]
    y_train = [[0,1,0,1,1]]
    hyperparameter_optimization(X_train, y_train)
    