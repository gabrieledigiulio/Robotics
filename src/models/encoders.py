import torch
import torch.nn as nn
from .vq import VectorQuantizer


class VAEVisualEncoder(nn.Module):
    """
    Encodes an input image into a continuous latent space representation 
    using a Variational Autoencoder (VAE) approach with convolutional layers.
    """

    def __init__(self, in_channels=3, latent_dim=64):
        """
        Args:
            in_channels: Number of channels in the input image.
            latent_dim: Dimensionality of the output latent space.
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.flatten_dim = 256 * 4 * 4
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from the latent distribution 
        allowing for backpropagation through the stochastic node.

        Args:
            mu: Mean of the latent distribution.
            logvar: Log-variance of the latent distribution.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Args:
            x: Input image tensor.

        Returns:
            A tuple containing the sampled latent vector (z), the mean (mu), 
            and the log-variance (logvar).
        """
        features = self.cnn(x)
        features_flat = torch.flatten(features, start_dim=1)
        
        mu = self.fc_mu(features_flat)
        logvar = self.fc_logvar(features_flat)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar


class VQVisualEncoder(nn.Module):
    """
    Encodes an input image into a Vector-Quantized (VQ) discrete latent space representation.
    """

    def __init__(self, in_channels=3, embedding_dim=4, num_embeddings=512, commitment_cost=0.25):
        """
        Args:
            in_channels: Number of channels in the input image.
            embedding_dim: Dimensionality of each embedding vector in the codebook.
            num_embeddings: Total number of discrete embeddings in the codebook.
            commitment_cost: Weighting factor for the commitment loss.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1)
        )
        
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        """
        Args:
            x: Input image tensor.

        Returns:
            A tuple containing the flattened quantized embeddings, the flattened 
            continuous embeddings, the vector quantization loss, and the perplexity.
        """
        features = self.cnn(x)
        
        quantized, vq_loss, perplexity = self.vq_layer(features)
        
        flat_quantized = torch.flatten(quantized, start_dim=1)
        flat_continuous = torch.flatten(features, start_dim=1)
        
        return flat_quantized, flat_continuous, vq_loss, perplexity


class TactileEncoder(nn.Module):
    """
    Encodes tactile sensor features into a continuous latent space representation 
    using a Multi-Layer Perceptron (MLP) mapping to a probability distribution.
    """

    def __init__(self, in_features: int = 20, hidden_dim: int = 64, latent_dim: int = 8):
        """
        Args:
            in_features: Number of input tactile sensor features.
            hidden_dim: Number of units in the hidden layers.
            latent_dim: Dimensionality of the output latent space.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tactile features tensor.

        Returns:
            A tuple containing the sampled latent vector (z), the mean (mu), 
            and the log-variance (logvar).
        """
        h = self.net(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        # Prevent numerical explosion during exponential calculation
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class ProprioEncoder(nn.Module):
    """
    Encodes proprioceptive sensor features into a continuous latent space representation 
    using a Multi-Layer Perceptron (MLP) mapping to a probability distribution.
    """

    def __init__(self, in_features: int = 20, hidden_dim: int = 64, latent_dim: int = 8):
        """
        Args:
            in_features: Number of input proprioceptive sensor features.
            hidden_dim: Number of units in the hidden layers.
            latent_dim: Dimensionality of the output latent space.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input proprioceptive features tensor.

        Returns:
            A tuple containing the sampled latent vector (z), the mean (mu), 
            and the log-variance (logvar).
        """
        h = self.net(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        # Prevent numerical explosion during exponential calculation
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar