import torch
import torch.nn as nn
from .encoders import VAEVisualEncoder, TactileEncoder, ProprioEncoder, VQVisualEncoder
from .decoders import VAEVisualDecoder, TactileDecoder, ProprioDecoder, VQVisualDecoder
from .dynamics import DynamicsMLP, DiffusionDynamics, FlowMatchingDynamics
from utils.weights_init import init_weights

class WorldModel(nn.Module):
    def __init__(self, img_channels: int = 3, img_latent_dim: int = 64,
                 tac_features: int = 20, tac_latent_dim: int = 8,
                 proprio_features: int = 20, proprio_latent_dim: int = 8,
                 action_dim: int = 3, hidden_dim: int = 256, dynamics_hidden_dim: int = 2048,
                 dynamics_type: str = "mlp",
                 latent_type: str = "vae",
                 vq_num_embeddings: int = 512,
                 vq_embedding_dim: int = 4,
                 vq_commitment_cost: float = 0.25,
                 diffusion_steps: int = 100,
                 diffusion_beta_start: float = 1e-4,
                 diffusion_beta_end:   float = 0.02):
        super().__init__()

        self.dynamics_type = dynamics_type
        self.latent_type = latent_type
        self.img_latent_dim = img_latent_dim
        self.tac_latent_dim = tac_latent_dim
        self.proprio_latent_dim = proprio_latent_dim
        total_latent_dim = img_latent_dim + tac_latent_dim + proprio_latent_dim

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
        elif dynamics_type == "diffusion":
            self.dynamics = DiffusionDynamics(
                latent_dim = total_latent_dim,
                action_dim = action_dim,
                hidden_dim = hidden_dim,
                n_steps    = diffusion_steps,
                beta_start = diffusion_beta_start,
                beta_end   = diffusion_beta_end,
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
                f"Tipo di dinamica '{dynamics_type}' non supportato. "
                "Valori validi: 'mlp' | 'diffusion' | 'flow_matching'"
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

    def _requantize_vq(self, img_part: torch.Tensor) -> torch.Tensor:
        """Forza il tensore spaziale continuo ad agganciarsi ai vettori del VQ Codebook."""
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
        if self.latent_type == "vqvae":
            z_img_t, vq_loss_img, perplexity_img = self.visual_encoder(img_t)
            mu_img_t = z_img_t
            logvar_img_t = torch.zeros_like(z_img_t)
        else:
            z_img_t, mu_img_t, logvar_img_t = self.visual_encoder(img_t)
            vq_loss_img = torch.tensor(0.0, device=img_t.device)
            perplexity_img = torch.tensor(0.0, device=img_t.device)
            
        z_tac_t, mu_tac_t, logvar_tac_t = self.tactile_encoder(tac_t)
        z_pos_t, mu_pos_t, logvar_pos_t = self.proprio_encoder(pos_t)
        
        z_t = torch.cat([z_img_t, z_tac_t, z_pos_t], dim=-1)
        mu_t = torch.cat([mu_img_t, mu_tac_t, mu_pos_t], dim=-1)

        if img_t1 is not None and tac_t1 is not None and pos_t1 is not None:
            img_pred_t = self.visual_decoder(mu_img_t)
            tac_pred_t = self.tactile_decoder(mu_tac_t)
            pos_pred_t = self.proprio_decoder(mu_pos_t)

            with torch.no_grad():
                if self.latent_type == "vqvae":
                    mu_img_t1, _, _ = self.visual_encoder(img_t1)
                else:
                    _, mu_img_t1, _ = self.visual_encoder(img_t1)
                _, mu_tac_t1, _ = self.tactile_encoder(tac_t1)
                _, mu_pos_t1, _ = self.proprio_encoder(pos_t1)
                
            z_t1_target = torch.cat([mu_img_t1, mu_tac_t1, mu_pos_t1], dim=-1)

            if self.dynamics_type in ["mlp"]:
                pred = self.dynamics(mu_t.detach(), action_t)
                target = z_t1_target
            else:
                pred, target = self.dynamics(mu_t.detach(), action_t, z_next=z_t1_target)

            return img_pred_t, tac_pred_t, pos_pred_t, mu_img_t, logvar_img_t, mu_tac_t, logvar_tac_t, mu_pos_t, logvar_pos_t, pred, target, vq_loss_img, perplexity_img

        else:
            z_next = self.dynamics.sample(mu_t, action_t) if hasattr(self.dynamics, 'sample') else self.dynamics(mu_t, action_t)
            
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
        if self.latent_type == "vqvae":
            mu_img_0, _, _ = self.visual_encoder(img_0)
        else:
            _, mu_img_0, _ = self.visual_encoder(img_0)
            
        _, mu_tac_0, _ = self.tactile_encoder(tac_0)
        _, mu_pos_0, _ = self.proprio_encoder(pos_0)
        
        z_curr = torch.cat([mu_img_0, mu_tac_0, mu_pos_0], dim=-1)
        
        _, N, _ = actions_sequence.shape
        pred_latents = []
        
        for step in range(N):
            action = actions_sequence[:, step, :]
            
            z_next = self.dynamics.sample(z_curr, action)
            
            # Re-quantize image part to snap to the codebook (prevents error accumulation)
            if self.latent_type == "vqvae":
                img_part = z_next[:, :self.img_latent_dim]
                quantized_flat = self._requantize_vq(img_part)
                z_next = torch.cat([quantized_flat, z_next[:, self.img_latent_dim:]], dim=-1)
            
            # Clamp continuous latents to prevent Euler integration explosion (inf/nan)
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