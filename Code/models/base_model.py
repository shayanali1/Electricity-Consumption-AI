"""
Base model class for electricity consumption prediction.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

class BaseModel(ABC):
    """
    Abstract base class for electricity consumption prediction models.
    """
    
    def __init__(self):
        """
        Initialize the base model.
        """
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
    
    @abstractmethod
    def build(self, **kwargs):
        """
        Build the model with specified parameters.
        
        Parameters:
        **kwargs: Model-specific parameters
        """
        pass
    
    def fit(self, X_train, y_train):
        """
        Train the model.
        
        Parameters:
        X_train: Training features
        y_train: Training target
        
        Returns:
        self: Trained model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() before fit().")
        
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        X: Features for prediction
        
        Returns:
        array: Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() before predict().")
        
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model performance.
        
        Parameters:
        y_true: True values
        y_pred: Predicted values
        
        Returns:
        dict: Performance metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"Root Mean Square Error (RMSE): {rmse:.4f} kWh")
        print(f"Mean Absolute Error (MAE): {mae:.4f} kWh")
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }
    
    def get_feature_importance(self):
        """
        Get feature importance for the model.
        
        Returns:
        array: Feature importance values
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not have feature_importances_ attribute")
        
        return self.model.feature_importances_
    
    def set_feature_names(self, feature_names):
        """
        Set feature names for the model.
        
        Parameters:
        feature_names: List of feature names
        """
        self.feature_names = feature_names
