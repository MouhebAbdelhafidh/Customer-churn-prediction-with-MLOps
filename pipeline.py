import pandas as pd     # Data preprocessiong operations
import inspect
import numpy as np     # Linear algebra operations
import csv     # CSV file operations 
import matplotlib.pyplot as plt     # Visualizations 
import seaborn as sns     # Visualizations 
import scipy.stats as stats 
from sklearn.preprocessing import LabelEncoder     # Encoding ordinal features
from sklearn.preprocessing import OneHotEncoder     # Encoding nominal features
from sklearn.feature_selection import SelectKBest, chi2     # Feature selection fun  ctions
from sklearn.preprocessing import StandardScaler     # Standrization
from sklearn.ensemble import RandomForestClassifier, StackingClassifier     # Random forest model and StackingClassifier lib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score    # Model evaluation metrics
from sklearn.model_selection import GridSearchCV     # Model hyperparameters grid
import optuna     # Hyperparameters fine-tuning
import logging     # Customizing fetch messages
from imblearn.over_sampling import SMOTE     # Handle dataset imbalance 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
from imblearn.over_sampling import ADASYN

# Data preparation
def prepare_data(training_dataset_path="training_dataset.csv",
		test_dataset_path="test_dataset.csv",
		replace_outliers_with_mean_columns,
		replace_outliers_with_median_columns):
    handle_outliers(training_dataset, replace_outliers_with_median_columns, replace_outliers_with_mean_columns)
# Handeling outliers
def compute_bounds(df, column):
    '''
    Helper function that calculates the bounds of a column
    
    df :               Dataframe name
    Q1 :               First quantile
    Q3 :               Third quantile
    IQR :              Quantile range
    lower_bound :      Floor of values
    upper_bound :      Ceil of values
    '''
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return lower_bound, upper_bound

def Replace_outliers_with_mean(df, column):
    '''
    Checks for outliers in a column and replaces them with the mean value
    
    df :               Dataframe name
    column:            Column name
    mean_value :       Mean value of the column
    '''
    
    lower_bound, upper_bound = compute_bounds(df, column)
    
    mean_value = df[column].mean()
    
    df[column] = df[column].where((df[column] >= lower_bound) & (df[column] <= upper_bound), mean_value)
    
    return df

def Replace_outliers_with_median(df, column):
    '''
    Checks for outliers in a column and replaces them with the median value
    
    df :               Dataframe name
    column:            Column name
    mean_value :       Median value of the column
    '''
    
    lower_bound, upper_bound = compute_bounds(df, column)
    
    median_value = df[column].median()
    
    df[column] = df[column].where((df[column] >= lower_bound) & (df[column] <= upper_bound), median_value)
    
    return df

# training_dataset_mode_columns = []
replace_outliers_with_mean_columns = ['Total day minutes', 
                                 'Total day charge', 
                                 'Total eve minutes', 
                                 'Total eve charge', 
                                 'Total night minutes', 
                                 'Total night charge', 
                                 'Total intl minutes', 
                                 'Total intl charge']

replace_outliers_with_median_columns = ['Account length',
                                   'Total day calls',
                                   'Total eve calls',
                                   'Total night calls',
                                   'Total intl calls',
                                   'Customer service calls']

def handle_outliers(df, median_cols, mean_cols):
    '''
    Changes outliers in the dataset using the adequat function
    
    df :                  Dataframe name
    median_cols :         Columns to replace their outliers with median
    mean_cols :           Columns to replace their outliers with mean
    '''
    for i in range(2): 
        for column in mean_cols:
            df = Replace_outliers_with_mean(df, column)
            print(f"mean value after iteration {i+1} of {column} : {df[column].mean()}")
        for column in median_cols:
            df = Replace_outliers_with_median(df, column)
            print(f"median value after iteration {i+1} of {column} : {df[column].mean()}")
    
    # Deleting rows with very large outliers        
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        lower_bound, upper_bound = compute_bounds(df, column)
        # Identify and drop rows with outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        df = df[~outliers]
    
    check_outliers(df)
