import torch
import torch.nn as nn

class VAEVisualEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.cnn(x)
        features_flat = torch.flatten(features, start_dim=1)
        
        mu = self.fc_mu(features_flat)
        logvar = self.fc_logvar(features_flat)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar