"""
Data loader module for processing database files.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class DatabaseDataset(Dataset):
    """PyTorch Dataset for loading and processing database files."""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray = None):
        """
        Initialize the dataset.
        
        Args:
            data: Feature data as numpy array
            targets: Target values (optional for inference)
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.data[idx], self.targets[idx]
        return self.data[idx]


class DataProcessor:
    """Handles loading and preprocessing of database files."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(file_path)
    
    def load_excel(self, file_path: str) -> pd.DataFrame:
        """Load data from Excel file."""
        return pd.read_excel(file_path)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame containing the loaded data
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return self.load_csv(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return self.load_excel(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def preprocess(self, df: pd.DataFrame, target_column: str = None, 
                   fit_scaler: bool = True) -> tuple:
        """
        Preprocess the data for neural network training.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column (if any)
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        # Separate features and target
        if target_column and target_column in df.columns:
            targets = df[target_column].values
            features_df = df.drop(columns=[target_column])
        else:
            targets = None
            features_df = df
        
        # Store feature columns for later use
        self.feature_columns = features_df.columns.tolist()
        
        # Handle numeric columns only
        numeric_df = features_df.select_dtypes(include=[np.number])
        
        # Scale features
        if fit_scaler:
            features = self.scaler.fit_transform(numeric_df.values)
        else:
            features = self.scaler.transform(numeric_df.values)
        
        return features, targets
    
    def create_dataloader(self, features: np.ndarray, targets: np.ndarray = None,
                          batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Create a PyTorch DataLoader from the processed data.
        
        Args:
            features: Feature array
            targets: Target array (optional)
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            PyTorch DataLoader
        """
        dataset = DatabaseDataset(features, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_data_loaders(data_path: str, target_column: str = None,
                     batch_size: int = 32, train_split: float = 0.8) -> tuple:
    """
    Convenience function to create train and validation data loaders.
    
    Args:
        data_path: Path to the data file
        target_column: Name of the target column
        batch_size: Batch size
        train_split: Proportion of data for training
        
    Returns:
        Tuple of (train_loader, val_loader, processor)
    """
    processor = DataProcessor()
    df = processor.load_data(data_path)
    
    # Split data
    train_size = int(len(df) * train_split)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Process training data
    train_features, train_targets = processor.preprocess(
        train_df, target_column=target_column, fit_scaler=True
    )
    
    # Process validation data
    val_features, val_targets = processor.preprocess(
        val_df, target_column=target_column, fit_scaler=False
    )
    
    # Create data loaders
    train_loader = processor.create_dataloader(
        train_features, train_targets, batch_size=batch_size, shuffle=True
    )
    val_loader = processor.create_dataloader(
        val_features, val_targets, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, processor
