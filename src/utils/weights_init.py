import torch.nn as nn


def init_weights(m):
    """
    Initializes the weights of a PyTorch module.
    
    Applies Kaiming Normal initialization to Linear, Conv2d, and 
    ConvTranspose2d layers (optimized for ReLU nonlinearity). 
    Initializes BatchNorm2d weights to 1 and biases to 0.
    
    Args:
        m: A PyTorch module (torch.nn.Module).
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)