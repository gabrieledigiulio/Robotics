import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModelLoss(nn.Module):
    """
    Computes the loss for the World Model, consisting of an image reconstruction 
    loss (MSE) and a Kullback-Leibler (KL) divergence penalty.
    """

    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Weighting factor for the KL divergence term.
        """
        super().__init__()
        self.beta = beta

    def forward(self, img_pred, img_target, mu, logvar):
        """
        Computes the forward pass for the World Model loss.

        Args:
            img_pred: The predicted or reconstructed image tensor.
            img_target: The ground truth image tensor.
            mu: The mean of the latent distribution.
            logvar: The log-variance of the latent distribution.

        Returns:
            A tuple containing the total loss tensor (for backpropagation) 
            and a dictionary with the individual scalar loss components (for logging).
        """
        batch_size = img_pred.shape[0]

        loss_img = F.mse_loss(img_pred, img_target, reduction="mean")

        kl_divergence = -0.5 * torch.mean(
            torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )

        total_loss = loss_img + (self.beta * kl_divergence)

        loss_dict = {
            "total_loss":     total_loss.item(),
            "recon_img_loss": loss_img.item(),
            "kl_loss":        kl_divergence.item(),
        }

        return total_loss, loss_dict
