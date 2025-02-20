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
import logging     # Customizing fetch messages
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
from imblearn.over_sampling import ADASYN
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Data preparation
def prepare_data(training_dataset_path="training_dataset.csv", test_dataset_path="test_dataset.csv"):
    # Columns to deal with
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

    categorical_columns = ['State', 'International plan', 'Voice mail plan', 'Churn']

    # Importing datasets
    training_dataset = pd.read_csv(training_dataset_path)
    test_dataset = pd.read_csv(test_dataset_path)

    # Handeling outliers
    handle_outliers(training_dataset, replace_outliers_with_median_columns, replace_outliers_with_mean_columns)
    handle_outliers(test_dataset, replace_outliers_with_median_columns, replace_outliers_with_mean_columns)

    # Encoding
    encoding_categorical_features(training_dataset, categorical_columns)
    encoding_categorical_features(test_dataset, categorical_columns)

    # Splitting the features from the tagrget
    X1 = training_dataset.drop(columns=['Churn'])  # All columns except target
    y1 = training_dataset['Churn']  # The target variable
    X2 = test_dataset.drop(columns=['Churn'])
    y2 = test_dataset['Churn']

    # Feature selection
    X1 = delete_correlated_features(X1, threshold=0.9)
    X2 = delete_correlated_features(X2, threshold=0.9)

    # Standarization
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.transform(X2)

    # Handeling imbalanced data
    adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)
    X1, y1 = adasyn.fit_resample(X1, y1)

    return X1, X2, y1, y2

# Modeling
def train_model(X1, y1):
    rf_model = random_forest_model(X1, y1)
    return rf_model

def save_model(model):
    joblib.dump(model, "model.joblib")

def load_model():
    model = joblib.load("model.joblib")
    return model

def evaluate_model(X2, y2):
    model = load_model()
    y_pred = model.predict(X2)
    accuracy = accuracy_score(y2, y_pred)
    print("Accuracy score:", accuracy * 100)
    model.predict([[18,	117.0,	408,	0,	0,	184.500000,	97.0,	203.355322,	80.0,	215.8,	90.0,	8.7,	4.0,	1.0]])
    matrix = confusion_matrix(y2, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")

# Helper functions

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
            #print(f"mean value after iteration {i+1} of {column} : {df[column].mean()}")
        for column in median_cols:
            df = Replace_outliers_with_median(df, column)
            #print(f"median value after iteration {i+1} of {column} : {df[column].mean()}")
    
    # Deleting rows with very large outliers        
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        lower_bound, upper_bound = compute_bounds(df, column)
        # Identify and drop rows with outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        df = df[~outliers]

# Encoding
def encoding_categorical_features(df, columns):
    '''
    Encodes ordinal categorical features
    
    df :                   Dataframe name
    columns :              Columns list
    label_encoder :        LabelEncoder instance
    '''
    label_encoder = LabelEncoder()
    for column in columns:
        # Encoding the categorical column
        df[column] = label_encoder.fit_transform(df[column])

# Feature selection
def delete_correlated_features(df, threshold):
    '''
    Deletes features with corellation higher than the threshold
    
    df :                          Original dataframe name
    threshold :                   Correlation threshold
    correlation_matrix :          Df correlation matrix
    upper_triangle :              Upper triangle of the correlation matrix
    to_drop :                     Columns to drop
    '''
    correlation_matrix = df.corr()

    # Find the upper triangle of the correlation matrix (we only need to check one half)
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    # Get a list of columns with correlations above the threshold
    to_drop = [column for column in upper_triangle.columns if any(abs(upper_triangle[column]) > threshold)]

    # Drop highly correlated features
    df = df.drop(columns=to_drop)
    
    return df

# Modeling
def random_forest_model(X, y):
    '''
    Train a Random Forest model on the provided data without splitting.

    X :               Features dataset.
    y :               Target labels.
    rf_model:            Forest model.
    '''
    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X, y)
    joblib.dump(rf_model, "random_forest_model.joblib")
    print("Model trained and saved as 'random_forest_model.joblib'.")
    return rf_model

