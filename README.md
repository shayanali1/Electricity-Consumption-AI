# Electricity Consumption AI

A machine learning project for predicting electricity consumption patterns.

## Overview

This project uses machine learning models to predict electricity consumption based on time features and weather data. It includes data generation, preprocessing, model training, evaluation, and visualization components.
## Features

- Synthetic data generation with realistic electricity consumption patterns
- Support for multiple machine learning models (Random Forest, XGBoost)
- Comprehensive data preprocessing pipeline
- Model evaluation with standard metrics (RMSE, MAE)
- Visualization of results and feature importance
- Prediction of future electricity consumption

## Models

- **Random Forest**: Ensemble learning method that operates by constructing multiple decision trees
- **XGBoost**: Optimized gradient boosting implementation with high performance

## Usage

### Basic Usage

```bash
python main.py
```

This will:
1. Generate synthetic electricity consumption data
2. Train a Random Forest model
3. Evaluate the model
4. Generate visualizations
5. Make predictions for the next 24 hours

### Command Line Options

```bash
python main.py --model xgboost --samples 17520 --test-size 0.25
```

Available options:
- `--model`: Model type to use (`random_forest` or `xgboost`)
- `--samples`: Number of samples to generate
- `--test-size`: Proportion of data to use for testing
- `--random-state`: Random seed for reproducibility
- `--no-plots`: Disable plotting

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

## Future Improvements

- Add support for real data import
- Implement more advanced models (LSTM, Transformer)
- Add hyperparameter tuning
- Create a web interface for predictions
- Integrate with smart home systems

## License

This project is licensed under the MIT License - see the LICENSE file for details.
