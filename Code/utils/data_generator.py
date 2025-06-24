"""
Synthetic data generation for electricity consumption prediction.
"""

import pandas as pd
import numpy as np
import datetime
from config.config import DATA_GENERATION

def generate_synthetic_data(n_samples=None, start_date=None, freq=None):
    """
    Generate synthetic electricity consumption data.

    Parameters:
    n_samples (int): Number of data points to generate
    start_date (str): Start date for the time series
    freq (str): Frequency of the time series

    Returns:
    pandas.DataFrame: Synthetic dataset with units:
        - temperature: °C (Celsius)
        - humidity: % (Percentage)
        - consumption: kWh (Kilowatt-hours)
    """
    # Use default values from config if not provided
    n_samples = n_samples or DATA_GENERATION['default_samples']
    start_date = start_date or DATA_GENERATION['start_date']
    freq = freq or DATA_GENERATION['frequency']

    # Handle 'now' as start date
    if start_date == 'now':
        start_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Create time index
    date_rng = pd.date_range(start=start_date, periods=n_samples, freq=freq)

    # Initialize DataFrame
    df = pd.DataFrame(date_rng, columns=['timestamp'])

    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Generate synthetic temperature data (seasonal pattern)
    avg_temp = 20  # Mean temperature in Celsius
    df['temperature'] = avg_temp + 10 * np.sin((df['timestamp'].dt.dayofyear * 2 * np.pi / 365) - np.pi/2) + np.random.normal(0, 3, n_samples)

    # Generate synthetic humidity data (correlated with temperature)
    df['humidity'] = 60 - 0.5 * (df['temperature'] - avg_temp) + np.random.normal(0, 10, n_samples)
    df['humidity'] = df['humidity'].clip(0, 100)  # Clip to valid humidity range

    # Generate synthetic electricity consumption (kWh)
    # Base load
    base_load = 0.8

    # Hourly pattern: higher during morning and evening
    hourly_pattern = np.array([0.6, 0.5, 0.4, 0.3, 0.3, 0.5, 0.8, 1.2, 1.4, 1.3, 1.2, 1.1,
                            1.2, 1.3, 1.2, 1.1, 1.0, 1.2, 1.5, 1.4, 1.3, 1.0, 0.8, 0.7])

    # Weekend effect: more consumption during day on weekends
    weekend_effect = 0.2

    # Seasonal effect: more consumption in extreme temperatures
    def temp_effect(temp):
        return 0.2 * ((temp - 20) / 10) ** 2

    # Calculate consumption
    df['consumption'] = base_load

    # Add hourly pattern
    df['consumption'] *= hourly_pattern[df['hour']]

    # Add weekend effect
    df.loc[df['weekend'] == 1, 'consumption'] += weekend_effect

    # Add temperature effect
    df['consumption'] += df['temperature'].apply(temp_effect)

    # Add random noise
    df['consumption'] += np.random.normal(0, 0.1, n_samples)

    # Ensure no negative values
    df['consumption'] = df['consumption'].clip(0.1)

    # Scale to realistic kWh values for a household
    df['consumption'] *= 1.5

    return df
