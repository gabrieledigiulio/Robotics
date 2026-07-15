import torch
import torch.nn as nn

class VAEVisualDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=3):
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
        x = self.fc(z)
        
        x = x.view(-1, 256, 4, 4)
        
        reconstructed_img = self.decoder(x)
        
        return reconstructed_img

class VQVisualDecoder(nn.Module):
    def __init__(self, embedding_dim=4, out_channels=3):
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
        x = z.view(-1, self.embedding_dim, 8, 8)
        x = self.pre_conv(x)
        reconstructed_img = self.decoder(x)
        return reconstructed_img

class TactileDecoder(nn.Module):
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64, out_features: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, z):
        return self.net(z)

class ProprioDecoder(nn.Module):
    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64, out_features: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, z):
        return self.net(z)