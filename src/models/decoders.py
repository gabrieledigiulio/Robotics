import torch
import torch.nn as nn


class VAEVisualDecoder(nn.Module):
    """
    Decodes a flat latent vector into an image reconstruction using 
    transposed convolutions.
    """

    def __init__(self, latent_dim=64, out_channels=3):
        """
        Args:
            latent_dim: Dimensionality of the input latent space.
            out_channels: Number of channels in the output image.
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            
            nn.Sigmoid() 
        )

    def forward(self, z):
        """
        Args:
            z: Input latent vector tensor.
            
        Returns:
            Reconstructed image tensor with values in the range [0, 1].
        """
        x = self.fc(z)
        
        x = x.view(-1, 256, 4, 4)
        
        reconstructed_img = self.decoder(x)
        
        return reconstructed_img


class VQVisualDecoder(nn.Module):
    """
    Decodes a Vector-Quantized (VQ) spatial latent representation into 
    an image reconstruction.
    """

    def __init__(self, embedding_dim=4, out_channels=3):
        """
        Args:
            embedding_dim: Dimensionality of the VQ embeddings (channels).
            out_channels: Number of channels in the output image.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.pre_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            
            nn.Sigmoid() 
        )

    def forward(self, z):
        """
        Args:
            z: Input quantized latent tensor.
            
        Returns:
            Reconstructed image tensor with values in the range [0, 1].
        """
        x = z.view(-1, self.embedding_dim, 8, 8)
        x = self.pre_conv(x)
        reconstructed_img = self.decoder(x)
        return reconstructed_img


class TactileDecoder(nn.Module):
    """
    Decodes a flat latent vector back into tactile sensor features 
    using a Multi-Layer Perceptron (MLP).
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64, out_features: int = 20):
        """
        Args:
            latent_dim: Dimensionality of the input latent space.
            hidden_dim: Number of units in the hidden layers.
            out_features: Number of output tactile features.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, z):
        """
        Args:
            z: Input latent vector tensor.
            
        Returns:
            Reconstructed tactile features.
        """
        return self.net(z)


class ProprioDecoder(nn.Module):
    """
    Decodes a flat latent vector back into proprioceptive sensor features 
    using a Multi-Layer Perceptron (MLP).
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64, out_features: int = 20):
        """
        Args:
            latent_dim: Dimensionality of the input latent space.
            hidden_dim: Number of units in the hidden layers.
            out_features: Number of output proprioceptive features.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, z):
        """
        Args:
            z: Input latent vector tensor.
            
        Returns:
            Reconstructed proprioceptive features.
        """
        return self.net(z)