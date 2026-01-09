"""
Utility functions for the deep learning project.
"""

import torch
import numpy as np
import random
from pathlib import Path
import json


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model, save_dir: str, filename: str = 'model.pt'):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model to save
        save_dir: Directory to save the model
        filename: Name of the checkpoint file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': getattr(model, 'input_dim', None),
            'hidden_dims': getattr(model, 'hidden_dims', None),
            'output_dim': getattr(model, 'output_dim', None),
        }
    }
    
    torch.save(checkpoint, save_path / filename)


def load_model(model, load_path: str):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        load_path: Path to the checkpoint file
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_dim: int):
    """Print a summary of the model architecture."""
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(model)
    print("-" * 50)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Input dimension: {input_dim}")
    print("=" * 50)
