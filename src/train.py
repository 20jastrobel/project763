"""
Training script for the neural network models.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json

from data_loader import get_data_loaders, DataProcessor
from model import create_model
from utils import save_model, load_model, EarlyStopping, set_seed


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_data in train_loader:
        if isinstance(batch_data, (list, tuple)):
            inputs, targets = batch_data
            targets = targets.to(device)
        else:
            inputs = batch_data
            targets = None
        
        inputs = inputs.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # Handle different output types
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        if targets is not None:
            loss = criterion(outputs.squeeze(), targets)
        else:
            # For autoencoder, use reconstruction loss
            loss = criterion(outputs, inputs)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            if isinstance(batch_data, (list, tuple)):
                inputs, targets = batch_data
                targets = targets.to(device)
            else:
                inputs = batch_data
                targets = None
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            if targets is not None:
                loss = criterion(outputs.squeeze(), targets)
            else:
                loss = criterion(outputs, inputs)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train(args):
    """Main training function."""
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    train_loader, val_loader, processor = get_data_loaders(
        args.data_path,
        target_column=args.target_column,
        batch_size=args.batch_size,
        train_split=args.train_split
    )
    
    # Get input dimension from data
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)):
        input_dim = sample_batch[0].shape[1]
    else:
        input_dim = sample_batch.shape[1]
    
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = create_model(
        args.model_type,
        input_dim=input_dim,
        output_dim=args.output_dim
    )
    model = model.to(device)
    print(f"Model: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, args.save_path, 'best_model.pt')
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model and history
    save_model(model, args.save_path, 'final_model.pt')
    
    history_path = Path(args.save_path) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # Print feature weights if applicable
    if hasattr(model, 'get_feature_weights'):
        weights = model.get_feature_weights()
        print("\nLearned Feature Weights:")
        for i, w in enumerate(weights):
            print(f"  Feature {i}: {w.item():.4f}")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train neural network for weight assignment')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data file')
    parser.add_argument('--target_column', type=str, default=None,
                        help='Name of the target column')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Training data split ratio')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='weight_assignment',
                        choices=['weight_assignment', 'autoencoder', 'attention'],
                        help='Type of model to train')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='Output dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--save_path', type=str, default='models',
                        help='Path to save models')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
