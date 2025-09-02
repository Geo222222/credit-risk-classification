"""
Utility functions and classes for the credit risk classification pipeline.
"""

import os
import yaml
import numpy as np
from pathlib import Path


class CostMatrix:
    """Class to handle cost-aware decision making."""
    
    def __init__(self, fn_cost=1000, fp_cost=100):
        """
        Initialize cost matrix.
        
        Args:
            fn_cost (float): Cost of False Negative (missing a default)
            fp_cost (float): Cost of False Positive (rejecting a good loan)
        """
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
    
    def expected_loss(self, y_true, y_pred):
        """
        Calculate expected loss based on predictions and true labels.
        
        Args:
            y_true (array): True binary labels
            y_pred (array): Predicted binary labels
            
        Returns:
            float: Expected loss
        """
        # False Negatives: predicted 0 (no default) but actual 1 (default)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        # False Positives: predicted 1 (default) but actual 0 (no default)
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        return fn * self.fn_cost + fp * self.fp_cost
    
    def optimal_threshold(self, y_true, y_proba):
        """
        Find optimal threshold that minimizes expected loss.
        
        Args:
            y_true (array): True binary labels
            y_proba (array): Predicted probabilities for positive class
            
        Returns:
            float: Optimal threshold
        """
        thresholds = np.linspace(0, 1, 101)
        losses = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            loss = self.expected_loss(y_true, y_pred)
            losses.append(loss)
        
        optimal_idx = np.argmin(losses)
        return thresholds[optimal_idx]


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    # Note: sklearn uses numpy's random state
