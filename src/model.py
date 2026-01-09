"""
Neural network model definitions for weight assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightAssignmentNetwork(nn.Module):
    """
    Neural network for assigning weights to database features.
    
    This model learns to assign importance weights to different features
    in the database through a multi-layer architecture.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = None, 
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize the network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output values
            dropout: Dropout rate for regularization
        """
        super(WeightAssignmentNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Feature importance weights (learnable)
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Apply learned feature weights
        weighted_x = x * F.softmax(self.feature_weights, dim=0)
        return self.network(weighted_x)
    
    def get_feature_weights(self) -> torch.Tensor:
        """
        Get the learned feature importance weights.
        
        Returns:
            Normalized feature weights
        """
        return F.softmax(self.feature_weights, dim=0).detach()


class AutoEncoder(nn.Module):
    """
    Autoencoder for learning compact representations of database features.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Number of input features
            latent_dim: Dimension of the latent space
        """
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed, latent)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get the latent representation."""
        return self.encoder(x)


class AttentionWeightNetwork(nn.Module):
    """
    Network using attention mechanism for feature weight assignment.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, num_heads: int = 4):
        """
        Initialize the attention-based network.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of outputs
            num_heads: Number of attention heads
        """
        super(AttentionWeightNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = 64
        
        # Feature embedding
        self.feature_embedding = nn.Linear(1, self.embed_dim)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = x.size(0)
        
        # Reshape for embedding: (batch, features, 1) -> (batch, features, embed_dim)
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        
        # Apply self-attention
        attended, attention_weights = self.attention(x, x, x)
        
        # Flatten and output
        attended = attended.view(batch_size, -1)
        output = self.fc(attended)
        
        return output, attention_weights


def create_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('weight_assignment', 'autoencoder', 'attention')
        input_dim: Number of input features
        **kwargs: Additional model arguments
        
    Returns:
        Initialized model
    """
    models = {
        'weight_assignment': WeightAssignmentNetwork,
        'autoencoder': AutoEncoder,
        'attention': AttentionWeightNetwork
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](input_dim, **kwargs)
