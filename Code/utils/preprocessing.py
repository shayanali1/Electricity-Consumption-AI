"""
Data preprocessing utilities for electricity consumption prediction.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.config import FEATURES, TRAINING

def preprocess_data(df):
    """
    Preprocess the data for model training.
    
    Parameters:
    df (pandas.DataFrame): Input dataset
    
    Returns:
    tuple: X and y for model training
    """
    # Create features and target
    time_features = FEATURES['time_features']
    weather_features = FEATURES['weather_features']
    all_features = time_features + weather_features
    target = FEATURES['target']
    
    X = df[all_features].copy()
    y = df[target].copy()
    
    return X, y

def split_and_scale_data(X, y, test_size=None, random_state=None):
    """
    Split the data into training and testing sets and scale the features.
    
    Parameters:
    X: Features
    y: Target
    test_size (float): Test set proportion
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: Scaled training and testing data, and scalers
    """
    # Use default values from config if not provided
    test_size = test_size or TRAINING['test_size']
    random_state = random_state or TRAINING['random_state']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Reshape y for scaling if needed
    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_test_reshaped = y_test.values.reshape(-1, 1)
    
    # We'll keep y unscaled for easier interpretation of results
    # But we'll return the scaler in case it's needed
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, scaler_y
