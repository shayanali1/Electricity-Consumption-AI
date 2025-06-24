"""
Visualization utilities for electricity consumption prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from config.config import VISUALIZATION

def plot_results(y_test, y_pred, title="Electricity Consumption Prediction", filename="prediction_results.png"):
    """
    Plot actual vs predicted values.

    Parameters:
    y_test: True values
    y_pred: Predicted values
    title (str): Plot title
    filename (str): Name of the file to save the plot
    """
    plt.figure(figsize=VISUALIZATION['figure_size'])

    # Plot actual and predicted values for a smaller window for clarity
    sample_size = min(VISUALIZATION['sample_size'], len(y_test))
    x_values = np.arange(sample_size)

    plt.plot(x_values, y_test[:sample_size], label='Actual', marker='o', alpha=0.7, markersize=4)

    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        # If y_pred is a 2D array, flatten it
        plt.plot(x_values, y_pred.flatten()[:sample_size], label='Predicted', marker='x', alpha=0.7, markersize=4)
    else:
        plt.plot(x_values, y_pred[:sample_size], label='Predicted', marker='x', alpha=0.7, markersize=4)

    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Electricity Consumption (kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add a text box with units information
    plt.figtext(0.02, 0.02, 'Units: kWh (kilowatt-hours)', fontsize=8, alpha=0.7)

    save_path = VISUALIZATION['save_path'] + filename
    plt.savefig(save_path)
    plt.close()
    print(f"Results plot saved to '{save_path}'")

def plot_feature_importance(model, feature_names, title="Feature Importance", filename="feature_importance.png"):
    """
    Plot feature importance for tree-based models.

    Parameters:
    model: Trained model with feature_importances_ attribute
    feature_names: Names of the features
    title (str): Plot title
    filename (str): Name of the file to save the plot
    """
    feature_importance = model.feature_importances_

    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)

    plt.figure(figsize=VISUALIZATION['figure_size'])
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()

    save_path = VISUALIZATION['save_path'] + filename
    plt.savefig(save_path)
    plt.close()
    print(f"Feature importance plot saved to '{save_path}'")

def plot_future_prediction(predictions, hours=24, title="Predicted Electricity Consumption", filename="future_prediction.png"):
    """
    Plot predictions for future time periods.

    Parameters:
    predictions: Predicted values
    hours (int): Number of hours to plot
    title (str): Plot title
    filename (str): Name of the file to save the plot
    """
    x_values = np.arange(hours)

    plt.figure(figsize=VISUALIZATION['figure_size'])
    plt.plot(x_values, predictions[:hours], marker='o')
    plt.title(title)
    plt.xlabel('Hour')
    plt.ylabel('Electricity Consumption (kWh)')
    plt.grid(True, alpha=0.3)
    plt.xticks(x_values)

    # Add total consumption annotation
    total_consumption = np.sum(predictions[:hours])
    plt.annotate(f'Total: {total_consumption:.2f} kWh/day',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

    # Add a text box with units information
    plt.figtext(0.02, 0.02, 'Units: kWh (kilowatt-hours)', fontsize=8, alpha=0.7)

    plt.tight_layout()

    save_path = VISUALIZATION['save_path'] + filename
    plt.savefig(save_path)
    plt.close()
    print(f"Future prediction plot saved to '{save_path}'")

def plot_consumption_patterns(data, column='consumption', by='hour', title=None, filename=None):
    """
    Plot consumption patterns by different time periods.

    Parameters:
    data (pandas.DataFrame): Dataset with consumption data
    column (str): Column name for consumption
    by (str): Time period to group by ('hour', 'day', 'month', 'day_of_week')
    title (str): Plot title
    filename (str): Name of the file to save the plot
    """
    if by not in ['hour', 'day', 'month', 'day_of_week']:
        raise ValueError("'by' must be one of 'hour', 'day', 'month', 'day_of_week'")

    # Group by the specified time period
    grouped = data.groupby(by)[column].mean()

    # Set default title and filename if not provided
    if title is None:
        title = f'Average Electricity Consumption by {by.capitalize()}'
    if filename is None:
        filename = f'consumption_by_{by}.png'

    plt.figure(figsize=VISUALIZATION['figure_size'])

    if by == 'hour':
        x_values = np.arange(24)
        plt.plot(x_values, grouped, marker='o')
        plt.xticks(x_values)
        plt.xlabel('Hour of Day')
    elif by == 'day':
        x_values = np.arange(1, 32)
        plt.plot(x_values, grouped, marker='o')
        plt.xlabel('Day of Month')
    elif by == 'month':
        x_values = np.arange(1, 13)
        plt.plot(x_values, grouped, marker='o')
        plt.xticks(x_values)
        plt.xlabel('Month')
    elif by == 'day_of_week':
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plt.bar(days, grouped)
        plt.xlabel('Day of Week')

    plt.title(title)
    plt.ylabel('Electricity Consumption (kWh)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = VISUALIZATION['save_path'] + filename
    plt.savefig(save_path)
    plt.close()
    print(f"Consumption pattern plot saved to '{save_path}'")
