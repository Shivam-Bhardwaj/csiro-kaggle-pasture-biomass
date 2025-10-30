"""
Training utilities and loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RMSELoss(nn.Module):
    """Root Mean Squared Error loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2))


class HuberLoss(nn.Module):
    """Huber loss for robust regression."""
    
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        residual = pred - target
        condition = torch.abs(residual) < self.delta
        squared_loss = 0.5 * residual ** 2
        linear_loss = self.delta * torch.abs(residual) - 0.5 * self.delta ** 2
        return torch.mean(torch.where(condition, squared_loss, linear_loss))


class CombinedLoss(nn.Module):
    """Combined loss for multi-task learning."""
    
    def __init__(self, weights=None):
        """
        Initialize combined loss.
        
        Args:
            weights: Dictionary of task weights, e.g., {'biomass': 1.0, 'clover': 0.5}
        """
        super().__init__()
        self.weights = weights or {}
        self.mse = nn.MSELoss()
        self.rmse = RMSELoss()
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss.
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of targets
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        for task_name, pred in predictions.items():
            if task_name in targets:
                task_weight = self.weights.get(task_name, 1.0)
                # Use RMSE for main task, MSE for others
                if task_name == 'biomass':
                    loss = self.rmse(pred, targets[task_name])
                else:
                    loss = self.mse(pred, targets[task_name])
                total_loss += task_weight * loss
        
        return total_loss


def calculate_metrics(predictions, targets):
    """
    Calculate regression metrics.
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        Dictionary of metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # max mode
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

