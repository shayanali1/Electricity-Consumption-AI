"""
Configuration settings for the Electricity Consumption AI project.
"""

# Data generation settings
DATA_GENERATION = {
    'default_samples': 8760,  # Default to 1 year of hourly data
    'start_date': 'now',  # Use current date as starting point
    'frequency': 'h',  # Hourly frequency (using 'h' instead of deprecated 'H')
}

# Model settings
MODEL_SETTINGS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Training settings
TRAINING = {
    'test_size': 0.2,
    'random_state': 42
}

# Visualization settings
VISUALIZATION = {
    'figure_size': (10, 6),
    'sample_size': 100,  # Number of samples to show in prediction plots
    'save_path': './output/'  # Path to save figures
}

# Feature settings
FEATURES = {
    'time_features': ['hour', 'day', 'month', 'day_of_week', 'weekend'],
    'weather_features': ['temperature', 'humidity'],
    'target': 'consumption'
}
