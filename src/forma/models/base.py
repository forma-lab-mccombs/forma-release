# Copyright 2026 The University of Texas at Austin.
#
# Licensed under The University of Texas at Austin Research License, Version
# 1.1 ("License"). You MAY NOT use this file except in compliance with the
# License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, AND THE
# UNIVERSITY OF TEXAS AT AUSTIN EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
# CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND
# WARRANTIES OF PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM
# COURSE OF DEALING OR USAGE OF TRADE. NO WARRANTY IS EITHER EXPRESS OR IMPLIED
# WITH RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no
# circumstances shall The University of Texas at Austin be liable for
# incidental, special, indirect, direct or consequential damages or loss of
# profits, interruption of business, or related expenses which may arise from
# use of Software or Documentation, including but not limited to those
# resulting from defects in Software and/or Documentation, or loss or
# inaccuracy of data of any kind.
#
# See the License for the specific language governing permissions and
# limitations under this License.

"""
Base model interface for all forecasting models.

This module defines the abstract BaseModel class that all models must inherit from
to ensure consistent API across different model types (sklearn, XGBoost, Forma, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    All models (sklearn wrappers, XGBoost, Forma, etc.) must inherit
    from this class and implement the fit() and predict() methods.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model.
        
        Args:
            model_name: Name identifier for this model
            **kwargs: Model-specific parameters
        """
        self.model_name = model_name
        self.hyperparameters = kwargs
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, train_data: Any, validation_data: Optional[Any] = None) -> None:
        """
        Train the model on the provided data.
        
        Args:
            train_data: Training dataset (format depends on model type)
            validation_data: Optional validation dataset
        """
        pass
    
    @abstractmethod
    def predict(self, test_data: Any) -> pd.DataFrame:
        """
        Generate predictions and save them to the specified path.
        
        This method is responsible for:
        1. Generating predictions from the test data
        2. Saving predictions in the standardized format to results/forecasts/
        
        Args:
            test_data: Test dataset (format depends on model type)

        Returns:
            DataFrame containing the predictions in the standardized format, which must have columns:
                'firm_id', 'target', 'quarter', 'forecast_horizon', 'prediction', 'model'
            "Tall" format where each row is a single forecasted value for a firm at a specific quarter and horizon
        """
        pass

    @abstractmethod
    def save(self, folder_path: str) -> None:
        """Save the trained model to the specified path."""
        pass

    @abstractmethod
    def load(self, folder_path: str) -> None:
        """Load a trained model from the specified path."""
        pass

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return the model's hyperparameters."""
        return self.hyperparameters.copy()
    
    def set_hyperparameters(self, **kwargs) -> None:
        """Update model hyperparameters."""
        self.hyperparameters.update(kwargs)
