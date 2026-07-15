import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class DynamicsMLP(nn.Module):
    def __init__(self, latent_dim=64, action_dim=3, hidden_dim=256):
        super().__init__()

        self.latent_dim = latent_dim
        input_dim = latent_dim + action_dim
        output_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z_curr, action_t, z_next=None):
        z_fuso = torch.cat([z_curr, action_t], dim=-1)
        return self.net(z_fuso)

    def sample(self, z_curr, action_t):
        """Uniform interface with Diffusion/FlowMatching dynamics."""
        return self.forward(z_curr, action_t)

    def rollout(self, z_0, actions_sequence):
        batch_size, N, _ = actions_sequence.shape

        pred_latents = []
        z_curr = z_0

        for t in range(N):
            action_t = actions_sequence[:, t, :]
            z_next_t = self.forward(z_curr, action_t)
            pred_latents.append(z_next_t)
            z_curr = z_next_t

        pred_latents = torch.stack(pred_latents, dim=1)
        return pred_latents


class DiffusionDynamics(nn.Module):

    def __init__(self, latent_dim: int = 64, action_dim: int = 3,
                 hidden_dim: int = 256, n_steps: int = 100,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_steps = n_steps

        betas               = torch.linspace(beta_start, beta_end, n_steps)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas",                         betas)
        self.register_buffer("alphas",                        alphas)
        self.register_buffer("alphas_cumprod",                alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",           alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",           torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        assert self.sqrt_one_minus_alphas_cumprod[-1] > 0.99, \
            "Schedule troppo debole: il segnale non viene distrutto abbastanza entro T step"
        
        self.register_buffer("posterior_variance",
                             betas * (1.0 - alphas_cumprod_prev) /
                             (1.0 - alphas_cumprod).clamp(min=1e-8))

        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        denoiser_in = latent_dim + hidden_dim + latent_dim + action_dim
        self.denoiser = nn.Sequential(
            nn.Linear(denoiser_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )


    def _sinusoidal_embed(self, timesteps: torch.Tensor) -> torch.Tensor:
        half  = self.latent_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) *
            torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


    def q_sample(self, z_clean: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(z_clean)
        sqrt_ab  = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_ab * z_clean + sqrt_1ab * noise


    def predict_noise(self, z_noisy: torch.Tensor, t: torch.Tensor,
                      z_cond: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        t_emb = self._sinusoidal_embed(t)
        t_emb = self.time_embed(t_emb)
        x = torch.cat([z_noisy, t_emb, z_cond, action_t], dim=-1)
        return self.denoiser(x)


    def forward(self, z_curr: torch.Tensor, action_t: torch.Tensor,
                z_next: torch.Tensor = None):
        if z_next is not None:
            B     = z_next.shape[0]
            t     = torch.randint(0, self.n_steps, (B,), device=z_next.device)
            noise = torch.randn_like(z_next)
            z_noisy    = self.q_sample(z_next, t, noise)
            noise_pred = self.predict_noise(z_noisy, t, z_curr, action_t)
            return noise_pred, noise
        else:
            return self.sample(z_curr, action_t)


    @torch.no_grad()
    def sample(self, z_cond: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        B, device = z_cond.shape[0], z_cond.device

        z = torch.randn(B, self.latent_dim, device=device)

        for i in reversed(range(self.n_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            noise_pred = self.predict_noise(z, t, z_cond, action_t)

            alpha_t    = self.alphas[i]
            beta_t     = self.betas[i]
            sqrt_1m_ab = self.sqrt_one_minus_alphas_cumprod[i]

            z = (1.0 / torch.sqrt(alpha_t)) * (
                z - (beta_t / sqrt_1m_ab) * noise_pred
            )

            if i > 0:
                z = z + torch.sqrt(self.posterior_variance[i]) * torch.randn_like(z)

        return z


    def rollout(self, z_img_0: torch.Tensor,
                actions_sequence: torch.Tensor) -> torch.Tensor:
        _, N, _ = actions_sequence.shape
        pred_img_latents = []
        z_curr = z_img_0

        for step in range(N):
            action  = actions_sequence[:, step, :]
            z_next  = self.sample(z_curr, action)
            pred_img_latents.append(z_next)
            z_curr  = z_next

        return torch.stack(pred_img_latents, dim=1)


class FlowMatchingDynamics(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, n_steps: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.n_steps    = n_steps
        self.hidden_dim = hidden_dim

        # Sinusoidal embed dim is independent of latent_dim
        # Using a fixed small projection keeps the param count lean
        sin_dim = min(hidden_dim, 128)  # at most 128-D sinusoidal features
        self.time_proj = nn.Sequential(
            nn.Linear(sin_dim, hidden_dim),
            nn.SiLU(),
        )
        self._sin_dim = sin_dim

        # 2 hidden layers instead of 3: enough for a small dataset
        vnet_in = latent_dim + hidden_dim + latent_dim + action_dim
        self.v_net = nn.Sequential(
            nn.Linear(vnet_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _sinusoidal_embed(self, timesteps: torch.Tensor) -> torch.Tensor:
        half  = self._sin_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) *
            torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = (timesteps[:, None].float() * 1000.0) * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def predict_velocity(self, z_t: torch.Tensor, t: torch.Tensor,
                         z_cond: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        t_emb = self._sinusoidal_embed(t)   # [B, sin_dim]
        t_emb = self.time_proj(t_emb)       # [B, hidden_dim]
        x = torch.cat([z_t, t_emb, z_cond, action_t], dim=-1)
        return self.v_net(x)

    def forward(self, z_curr: torch.Tensor, action_t: torch.Tensor,
                z_next: torch.Tensor = None):
        if z_next is not None:
            B = z_next.shape[0]
            device = z_next.device
            
            t = torch.rand(B, 1, device=device)
            
            z_0 = torch.randn_like(z_next)
            z_1 = z_next
            z_t = (1 - t) * z_0 + t * z_1
            
            v_true = z_1 - z_0
            
            v_pred = self.predict_velocity(z_t, t.squeeze(-1), z_curr, action_t)
            
            return v_pred, v_true
        else:
            return self.sample(z_curr, action_t)

    @torch.no_grad()
    def sample(self, z_cond: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        B, device = z_cond.shape[0], z_cond.device
        
        z = torch.randn(B, self.latent_dim, device=device)
        
        dt = 1.0 / self.n_steps
        
        for i in range(self.n_steps):
            t = torch.full((B,), i * dt, device=device, dtype=torch.float32)
            
            v_pred = self.predict_velocity(z, t, z_cond, action_t)
            
            z = z + v_pred * dt
            
        return z

    def rollout(self, z_img_0: torch.Tensor,
                actions_sequence: torch.Tensor) -> torch.Tensor:
        _, N, _ = actions_sequence.shape
        pred_img_latents = []
        z_curr = z_img_0

        for step in range(N):
            action = actions_sequence[:, step, :]
            z_next = self.sample(z_curr, action)
            pred_img_latents.append(z_next)
            z_curr = z_next

        return torch.stack(pred_img_latents, dim=1)