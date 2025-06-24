"""
Main entry point for the Electricity Consumption AI project.
"""

import argparse
import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_generator import generate_synthetic_data
from utils.preprocessing import preprocess_data, split_and_scale_data
from utils.visualization import (
    plot_results,
    plot_feature_importance,
    plot_future_prediction,
    plot_consumption_patterns
)
from models import RandomForestModel, XGBoostModel

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Electricity Consumption Prediction')

    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost'],
                        help='Model type to use (default: random_forest)')

    parser.add_argument('--samples', type=int, default=8760,
                        help='Number of samples to generate (default: 8760 - 1 year hourly)')

    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')

    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')

    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plotting')

    return parser.parse_args()

def main():
    """
    Main function to run the electricity consumption prediction pipeline.
    """
    # Parse arguments
    args = parse_arguments()

    # Generate synthetic data
    print("Generating synthetic electricity consumption data...")
    data = generate_synthetic_data(n_samples=args.samples)

    # Display data overview with units
    print("\nData overview:")
    # Create a copy of the data for display
    display_data = data.head().copy()

    # Add units to column names
    display_data = display_data.rename(columns={
        'temperature': 'temperature (°C)',
        'humidity': 'humidity (%)',
        'consumption': 'consumption (kWh)'
    })

    print(display_data)
    print(f"\nDataset shape: {data.shape}")
    print("\nUnits:")
    print("- temperature: °C (Celsius)")
    print("- humidity: % (Percentage)")
    print("- consumption: kWh (Kilowatt-hours)")

    # Preprocess data
    X, y = preprocess_data(data)

    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, _ = split_and_scale_data(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Create and train model
    if args.model == 'random_forest':
        print("\nTraining the Random Forest model...")
        model = RandomForestModel()
    else:
        print("\nTraining the XGBoost model...")
        model = XGBoostModel()

    # Build and train the model
    model.build(random_state=args.random_state)
    model.set_feature_names(X.columns)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate model
    print("\nModel evaluation:")
    metrics = model.evaluate(y_test, y_pred)

    # Visualizations
    if not args.no_plots:
        # Plot results
        plot_results(y_test, y_pred,
                    title=f"{args.model.replace('_', ' ').title()} Model Predictions")

        # Plot feature importance
        plot_feature_importance(model.model, X.columns,
                               title=f"Feature Importance - {args.model.replace('_', ' ').title()}")

        # Plot consumption patterns
        plot_consumption_patterns(data, by='hour',
                                 title='Average Hourly Electricity Consumption')
        plot_consumption_patterns(data, by='day_of_week',
                                 title='Average Electricity Consumption by Day of Week')

        # Predict for next day (24 hours)
        print("\nPredicting electricity consumption for the next 24 hours...")
        # For demonstration, we'll use the last 24 hours of features from our dataset
        next_day_features = X.iloc[-24:].copy()
        next_day_features_scaled = scaler_X.transform(next_day_features)
        predicted_consumption = model.predict(next_day_features_scaled)

        # Plot future predictions
        plot_future_prediction(predicted_consumption,
                              title=f"Predicted Electricity Consumption for Next 24 Hours")

        # Display prediction statistics with units
        print("\nPrediction statistics:")
        print(f"Average predicted consumption: {np.mean(predicted_consumption):.4f} kWh")
        print(f"Peak predicted consumption: {np.max(predicted_consumption):.4f} kWh")
        print(f"Minimum predicted consumption: {np.min(predicted_consumption):.4f} kWh")

        # Calculate daily consumption
        daily_consumption = np.sum(predicted_consumption)
        print(f"Total daily consumption: {daily_consumption:.2f} kWh")

        # Calculate estimated monthly consumption (based on daily average)
        monthly_consumption = daily_consumption * 30
        print(f"Estimated monthly consumption: {monthly_consumption:.2f} kWh")

        # Calculate estimated cost (assuming average electricity price of $0.15 per kWh)
        daily_cost = daily_consumption * 0.15
        monthly_cost = monthly_consumption * 0.15
        print(f"Estimated daily cost: ${daily_cost:.2f}")
        print(f"Estimated monthly cost: ${monthly_cost:.2f}")

if __name__ == "__main__":
    main()
