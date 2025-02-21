import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline import prepare_data
import numpy as np


def test_data_preparation():
    X_train, X_test, y_train, y_test = prepare_data()
    assert X_train.shape[0] > 0, "Training data should not be empty"
    assert X_test.shape[0] > 0, "Test data should not be empty"
    assert y_train.isnull().sum() == 0, "Training labels should not have missing values"
    assert y_test.isnull().sum() == 0, "Test labels should not have missing values"
