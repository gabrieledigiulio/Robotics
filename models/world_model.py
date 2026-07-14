import torch
import torch.nn as nn
from .encoders import VAEVisualEncoder
from .decoders import VAEVisualDecoder
from .dynamics import DynamicsMLP, DiffusionDynamics, FlowMatchingDynamics
from utils.weights_init import init_weights


class WorldModel(nn.Module):
    def __init__(self, img_channels: int = 3, img_latent_dim: int = 64,
                 action_dim: int = 3, hidden_dim: int = 256,
                 dynamics_type: str = "mlp",
                 diffusion_steps: int = 100,
                 diffusion_beta_start: float = 1e-4,
                 diffusion_beta_end:   float = 0.02):
        super().__init__()

        self.dynamics_type = dynamics_type

        self.visual_encoder = VAEVisualEncoder(
            in_channels=img_channels, latent_dim=img_latent_dim
        )

        if dynamics_type == "mlp":
            self.dynamics = DynamicsMLP(
                img_dim    = img_latent_dim,
                action_dim = action_dim,
                hidden_dim = hidden_dim,
            )
        elif dynamics_type == "diffusion":
            self.dynamics = DiffusionDynamics(
                img_dim    = img_latent_dim,
                action_dim = action_dim,
                hidden_dim = hidden_dim,
                n_steps    = diffusion_steps,
                beta_start = diffusion_beta_start,
                beta_end   = diffusion_beta_end,
            )
        elif dynamics_type == "flow_matching":
            from config import FLOW_MATCHING_STEPS
            self.dynamics = FlowMatchingDynamics(
                img_dim    = img_latent_dim,
                action_dim = action_dim,
                hidden_dim = hidden_dim,
                n_steps    = FLOW_MATCHING_STEPS,
            )
        else:
            raise ValueError(
                f"dynamics_type sconosciuto: {dynamics_type!r}. "
                "Valori validi: 'mlp' | 'diffusion' | 'flow_matching'"
            )

        self.visual_decoder = VAEVisualDecoder(
            latent_dim=img_latent_dim, out_channels=img_channels
        )

        self.apply(init_weights)


    def forward(self, img_t: torch.Tensor, action_t: torch.Tensor,
                img_t1: torch.Tensor = None):
        z_t, mu, logvar = self.visual_encoder(img_t)

        if self.dynamics_type == "mlp":
            z_next   = self.dynamics(z_t, action_t)
            img_pred = self.visual_decoder(z_next)
            return img_pred, mu, logvar, None, None

        if img_t1 is not None:
            img_pred_t = self.visual_decoder(mu)

            with torch.no_grad():
                _, mu_t1, _ = self.visual_encoder(img_t1)
            z_t1_target = mu_t1

            pred, target = self.dynamics(mu.detach(), action_t, z_next=z_t1_target)

            return img_pred_t, mu, logvar, pred, target

        else:
            z_next   = self.dynamics(mu, action_t)
            img_pred = self.visual_decoder(z_next)
            return img_pred, mu, logvar, None, None


    def rollout(self, img_0: torch.Tensor,
                actions_sequence: torch.Tensor) -> torch.Tensor:
        _, mu_0, _ = self.visual_encoder(img_0)

        pred_img_latents = self.dynamics.rollout(mu_0, actions_sequence)

        B, N, _ = pred_img_latents.shape
        z_flat   = pred_img_latents.view(B * N, -1)
        imgs_flat = self.visual_decoder(z_flat)

        _, C, H, W = imgs_flat.shape
        return imgs_flat.view(B, N, C, H, W)