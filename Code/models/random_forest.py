"""
Random Forest model for electricity consumption prediction.
"""

from sklearn.ensemble import RandomForestRegressor
from models.base_model import BaseModel
from config.config import MODEL_SETTINGS

class RandomForestModel(BaseModel):
    """
    Random Forest model for electricity consumption prediction.
    """
    
    def __init__(self):
        """
        Initialize the Random Forest model.
        """
        super().__init__()
    
    def build(self, **kwargs):
        """
        Build the Random Forest model with specified parameters.
        
        Parameters:
        **kwargs: Model-specific parameters that override the defaults
        
        Returns:
        self: Model instance
        """
        # Get default parameters from config
        params = MODEL_SETTINGS['random_forest'].copy()
        
        # Override with any provided parameters
        params.update(kwargs)
        
        # Build the model
        self.model = RandomForestRegressor(**params)
        
        return self
