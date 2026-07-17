import torch
import torch.nn as nn
from .encoders import VAEVisualEncoder, TactileEncoder, ProprioEncoder, VQVisualEncoder
from .decoders import VAEVisualDecoder, TactileDecoder, ProprioDecoder, VQVisualDecoder
from .dynamics import DynamicsMLP, FlowMatchingDynamics
from utils.weights_init import init_weights


class LatentScaler(nn.Module):
    """
    Normalizes the latent space using an Exponential Moving Average (EMA) 
    of the batch statistics to stabilize dynamics training.
    """

    def __init__(self, dim, momentum=0.01):
        """
        Args:
            dim: Dimensionality of the latent space to scale.
            momentum: EMA momentum for updating the running mean and variance.
        """
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.momentum = momentum
        self.is_fitted = False
        self.freeze = False
        
    def update(self, x):
        """
        Updates the running mean and variance using the current batch statistics.
        
        Args:
            x: Input latent tensor.
        """
        if self.training and not self.freeze:
            batch_mean = x.mean(dim=0).detach()
            batch_var = x.var(dim=0, unbiased=False).detach()
            if not self.is_fitted:
                self.mean.copy_(batch_mean)
                self.var.copy_(batch_var)
                self.is_fitted = True
            else:
                self.mean.lerp_(batch_mean, self.momentum)
                self.var.lerp_(batch_var, self.momentum)
                
    def forward(self, x):
        """
        Standardizes the input latent tensor.
        
        Args:
            x: Input latent tensor.
            
        Returns:
            Scaled latent tensor with zero mean and unit variance.
        """
        if self.training:
            self.update(x)
        std = torch.sqrt(self.var + 1e-6)
        return (x - self.mean) / std
        
    def inverse(self, x):
        """
        Reverts the standardization process to restore the original latent scale.
        
        Args:
            x: Scaled latent tensor.
            
        Returns:
            Unscaled latent tensor in the original distribution.
        """
        std = torch.sqrt(self.var + 1e-6)
        return x * std + self.mean


class WorldModel(nn.Module):
    """
    Core architecture integrating visual, tactile, and proprioceptive modalities 
    with a latent dynamics model.
    """

    def __init__(self, img_channels: int = 3, img_latent_dim: int = 64,
                 tac_features: int = 20, tac_latent_dim: int = 8,
                 proprio_features: int = 20, proprio_latent_dim: int = 8,
                 action_dim: int = 3, hidden_dim: int = 256, dynamics_hidden_dim: int = 2048,
                 dynamics_type: str = "mlp",
                 latent_type: str = "vae",
                 vq_num_embeddings: int = 512,
                 vq_embedding_dim: int = 4,
                 vq_commitment_cost: float = 0.25):
        """
        Args:
            img_channels: Number of input image channels.
            img_latent_dim: Dimensionality of the image latent representation.
            tac_features: Number of input tactile features.
            tac_latent_dim: Dimensionality of the tactile latent representation.
            proprio_features: Number of input proprioceptive features.
            proprio_latent_dim: Dimensionality of the proprioceptive latent representation.
            action_dim: Dimensionality of the action vector.
            hidden_dim: Hidden dimension size for the base MLP layers.
            dynamics_hidden_dim: Hidden dimension size specifically for the dynamics model.
            dynamics_type: Type of dynamics transition model ("mlp" or "flow_matching").
            latent_type: Type of latent space modeling ("vae" or "vqvae").
            vq_num_embeddings: Codebook size if using VQ-VAE.
            vq_embedding_dim: Embedding dimension if using VQ-VAE.
            vq_commitment_cost: Commitment cost factor if using VQ-VAE.
        """
        super().__init__()

        self.dynamics_type = dynamics_type
        self.latent_type = latent_type
        self.tac_latent_dim = tac_latent_dim
        self.proprio_latent_dim = proprio_latent_dim

        if self.latent_type == "vqvae":
            self.img_latent_dim = vq_embedding_dim * 8 * 8
        else:
            self.img_latent_dim = img_latent_dim

        total_latent_dim = self.img_latent_dim + tac_latent_dim + proprio_latent_dim
        
        self.scaler = LatentScaler(total_latent_dim)

        if self.latent_type == "vqvae":
            self.visual_encoder = VQVisualEncoder(
                in_channels=img_channels, 
                embedding_dim=vq_embedding_dim, 
                num_embeddings=vq_num_embeddings, 
                commitment_cost=vq_commitment_cost
            )
        else:
            self.visual_encoder = VAEVisualEncoder(
                in_channels=img_channels, latent_dim=img_latent_dim
            )
        self.tactile_encoder = TactileEncoder(
            in_features=tac_features, latent_dim=tac_latent_dim
        )
        self.proprio_encoder = ProprioEncoder(
            in_features=proprio_features, latent_dim=proprio_latent_dim
        )

        if dynamics_type == "mlp":
            self.dynamics = DynamicsMLP(
                latent_dim = total_latent_dim,
                action_dim = action_dim,
                hidden_dim = hidden_dim,
            )

        elif dynamics_type == "flow_matching":
            from config import FLOW_MATCHING_STEPS
            self.dynamics = FlowMatchingDynamics(
                latent_dim = total_latent_dim,
                action_dim = action_dim,
                hidden_dim = dynamics_hidden_dim,
                n_steps    = FLOW_MATCHING_STEPS,
            )
        else:
            raise ValueError(
                f"Dynamics type '{dynamics_type}' not supported. "
                "Valid values: 'mlp' | 'flow_matching'"
            )

        if self.latent_type == "vqvae":
            self.visual_decoder = VQVisualDecoder(
                embedding_dim=vq_embedding_dim, out_channels=img_channels
            )
        else:
            self.visual_decoder = VAEVisualDecoder(
                latent_dim=img_latent_dim, out_channels=img_channels
            )
        self.tactile_decoder = TactileDecoder(
            latent_dim=tac_latent_dim, out_features=tac_features
        )
        self.proprio_decoder = ProprioDecoder(
            latent_dim=proprio_latent_dim, out_features=proprio_features
        )

        self.apply(init_weights)

    def _scale(self, x):
        """Scales the input tensor using the internal LatentScaler."""
        return self.scaler(x) if self.latent_type == "vqvae" else x

    def _unscale(self, x):
        """Inverts the scaling of the input tensor."""
        return self.scaler.inverse(x) if self.latent_type == "vqvae" else x

    def _requantize_vq(self, img_part: torch.Tensor) -> torch.Tensor:
        """
        Forces the predicted continuous spatial tensor to snap back 
        to the discrete Vector Quantized (VQ) Codebook vectors.
        
        Args:
            img_part: Continuous image latent representation.
            
        Returns:
            Quantized and flattened latent representation.
        """
        vq = self.visual_encoder.vq_layer
        emb_dim = vq._embedding_dim
        weight = vq._embedding
        
        B_curr = img_part.shape[0]
        spatial_dim = int((self.img_latent_dim // emb_dim) ** 0.5)
        
        img_part_bchw = img_part.view(B_curr, emb_dim, spatial_dim, spatial_dim)
        img_part_bhwc = img_part_bchw.permute(0, 2, 3, 1).contiguous()
        flat_input = img_part_bhwc.view(-1, emb_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], weight.shape[0], device=img_part.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized_bhwc = torch.matmul(encodings, weight).view(B_curr, spatial_dim, spatial_dim, emb_dim)
        quantized_bchw = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return quantized_bchw.view(B_curr, -1)

    def forward(self, img_t: torch.Tensor, tac_t: torch.Tensor, pos_t: torch.Tensor, action_t: torch.Tensor,
                img_t1: torch.Tensor = None, tac_t1: torch.Tensor = None, pos_t1: torch.Tensor = None):
        """
        Performs a forward pass for either representation learning (if target step t+1 is None) 
        or dynamics modeling (if targets are provided).
        
        Args:
            img_t: Image tensor at step t.
            tac_t: Tactile tensor at step t.
            pos_t: Proprioception tensor at step t.
            action_t: Action tensor at step t.
            img_t1: Optional target image tensor at step t+1.
            tac_t1: Optional target tactile tensor at step t+1.
            pos_t1: Optional target proprioception tensor at step t+1.
            
        Returns:
            A tuple containing predictions, latent statistics, dynamics targets, 
            VQ losses, and perplexity scores.
        """
        if self.latent_type == "vqvae":
            z_img_t, mu_img_t, vq_loss_img, perplexity_img = self.visual_encoder(img_t)
            logvar_img_t = torch.zeros_like(mu_img_t)
        else:
            z_img_t, mu_img_t, logvar_img_t = self.visual_encoder(img_t)
            vq_loss_img = torch.tensor(0.0, device=img_t.device)
            perplexity_img = torch.tensor(0.0, device=img_t.device)
            
        z_tac_t, mu_tac_t, logvar_tac_t = self.tactile_encoder(tac_t)
        z_pos_t, mu_pos_t, logvar_pos_t = self.proprio_encoder(pos_t)
        
        z_t = torch.cat([z_img_t, z_tac_t, z_pos_t], dim=-1)
        mu_t = torch.cat([mu_img_t, mu_tac_t, mu_pos_t], dim=-1)
        
        if self.latent_type == "vqvae":
            dyn_input_t = torch.cat([z_img_t, mu_tac_t, mu_pos_t], dim=-1)
        else:
            dyn_input_t = mu_t

        if img_t1 is not None and tac_t1 is not None and pos_t1 is not None:
            if self.latent_type == "vqvae":
                img_pred_t = self.visual_decoder(z_img_t)
            else:
                img_pred_t = self.visual_decoder(mu_img_t)
            tac_pred_t = self.tactile_decoder(mu_tac_t)
            pos_pred_t = self.proprio_decoder(mu_pos_t)

            with torch.no_grad():
                if self.latent_type == "vqvae":
                    _, mu_img_t1, _, _ = self.visual_encoder(img_t1)
                else:
                    _, mu_img_t1, _ = self.visual_encoder(img_t1)
                _, mu_tac_t1, _ = self.tactile_encoder(tac_t1)
                _, mu_pos_t1, _ = self.proprio_encoder(pos_t1)
                
            z_t1_target = torch.cat([mu_img_t1, mu_tac_t1, mu_pos_t1], dim=-1)

            if self.dynamics_type in ["mlp"]:
                pred = self.dynamics(self._scale(dyn_input_t.detach()), action_t)
                target = self._scale(z_t1_target)
            else:
                pred, target = self.dynamics(self._scale(dyn_input_t.detach()), action_t, z_next=self._scale(z_t1_target))

            return img_pred_t, tac_pred_t, pos_pred_t, mu_img_t, logvar_img_t, mu_tac_t, logvar_tac_t, mu_pos_t, logvar_pos_t, pred, target, vq_loss_img, perplexity_img

        else:
            z_curr_scaled = self._scale(dyn_input_t)
            z_next_scaled = self.dynamics.sample(z_curr_scaled, action_t) if hasattr(self.dynamics, 'sample') else self.dynamics(z_curr_scaled, action_t)
            z_next = self._unscale(z_next_scaled)
            
            z_next_img = z_next[:, :self.img_latent_dim]
            
            if self.latent_type == "vqvae":
                z_next_img = self._requantize_vq(z_next_img)
                
            tac_end = self.img_latent_dim + self.tac_latent_dim
            z_next_tac = z_next[:, self.img_latent_dim:tac_end]
            z_next_pos = z_next[:, tac_end:]
            
            img_pred = self.visual_decoder(z_next_img)
            tac_pred = self.tactile_decoder(z_next_tac)
            pos_pred = self.proprio_decoder(z_next_pos)
            return img_pred, tac_pred, pos_pred, mu_img_t, logvar_img_t, mu_tac_t, logvar_tac_t, mu_pos_t, logvar_pos_t, None, None, vq_loss_img, perplexity_img

    def rollout(self, img_0: torch.Tensor, tac_0: torch.Tensor, pos_0: torch.Tensor,
                actions_sequence: torch.Tensor):
        """
        Autoregressively rolls out predictions over multiple future steps.
        
        Args:
            img_0: Initial image tensor.
            tac_0: Initial tactile tensor.
            pos_0: Initial proprioception tensor.
            actions_sequence: Sequence of future actions to condition the rollout.
            
        Returns:
            A tuple containing predicted multi-step image, tactile, and proprioceptive sequences.
        """
        if self.latent_type == "vqvae":
            z_img_0, _, _, _ = self.visual_encoder(img_0)
            z_curr_img = z_img_0
        else:
            _, z_curr_img, _ = self.visual_encoder(img_0)
            
        _, mu_tac_0, _ = self.tactile_encoder(tac_0)
        _, mu_pos_0, _ = self.proprio_encoder(pos_0)
        
        z_curr = torch.cat([z_curr_img, mu_tac_0, mu_pos_0], dim=-1)
        
        _, N, _ = actions_sequence.shape
        pred_latents = []
        
        for step in range(N):
            action = actions_sequence[:, step, :]
            
            z_curr_scaled = self._scale(z_curr)
            z_next_scaled = self.dynamics.sample(z_curr_scaled, action) if hasattr(self.dynamics, 'sample') else self.dynamics(z_curr_scaled, action)
            z_next = self._unscale(z_next_scaled)
            
            if self.latent_type == "vqvae":
                img_part = z_next[:, :self.img_latent_dim]
                quantized_flat = self._requantize_vq(img_part)
                z_next = torch.cat([quantized_flat, z_next[:, self.img_latent_dim:]], dim=-1)
            
            z_next = torch.clamp(z_next, min=-50.0, max=50.0)
            
            pred_latents.append(z_next)
            z_curr = z_next
            
        pred_latents = torch.stack(pred_latents, dim=1)

        B, N, _ = pred_latents.shape
        z_flat   = pred_latents.view(B * N, -1)
        
        z_flat_img = z_flat[:, :self.img_latent_dim]
        tac_end = self.img_latent_dim + self.tac_latent_dim
        z_flat_tac = z_flat[:, self.img_latent_dim:tac_end]
        z_flat_pos = z_flat[:, tac_end:]
        
        imgs_flat = self.visual_decoder(z_flat_img)
        tacs_flat = self.tactile_decoder(z_flat_tac)
        pos_flat  = self.proprio_decoder(z_flat_pos)

        _, C, H, W = imgs_flat.shape
        return imgs_flat.view(B, N, C, H, W), tacs_flat.view(B, N, -1), pos_flat.view(B, N, -1)