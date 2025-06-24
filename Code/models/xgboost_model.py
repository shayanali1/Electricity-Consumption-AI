"""
XGBoost model for electricity consumption prediction.
"""

from xgboost import XGBRegressor
from models.base_model import BaseModel
from config.config import MODEL_SETTINGS

class XGBoostModel(BaseModel):
    """
    XGBoost model for electricity consumption prediction.
    """
    
    def __init__(self):
        """
        Initialize the XGBoost model.
        """
        super().__init__()
    
    def build(self, **kwargs):
        """
        Build the XGBoost model with specified parameters.
        
        Parameters:
        **kwargs: Model-specific parameters that override the defaults
        
        Returns:
        self: Model instance
        """
        # Get default parameters from config
        params = MODEL_SETTINGS['xgboost'].copy()
        
        # Override with any provided parameters
        params.update(kwargs)
        
        # Build the model
        self.model = XGBRegressor(**params)
        
        return self
