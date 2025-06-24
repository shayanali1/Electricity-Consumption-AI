"""
Models package for Electricity Consumption AI.
"""

from models.base_model import BaseModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel

__all__ = ['BaseModel', 'RandomForestModel', 'XGBoostModel']
