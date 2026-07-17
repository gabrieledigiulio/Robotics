import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) base model that handles deterministic transitions 
    in the latent state space conditioned on actions.
    """

    def __init__(self, latent_dim=64, action_dim=3, hidden_dim=256):
        """
        Args:
            latent_dim: Dimensionality of the latent state space.
            action_dim: Dimensionality of the action vector.
            hidden_dim: Number of hidden units in the linear layers.
        """
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
        """
        Computes the single-step forward dynamics transition.

        Args:
            z_curr: Current latent state tensor.
            action_t: Action tensor applied at the current step.
            z_next: Optional target next latent state (ignored, kept for interface unified with FlowMatching).

        Returns:
            The predicted next latent state tensor.
        """
        z_fuso = torch.cat([z_curr, action_t], dim=-1)
        return self.net(z_fuso)

    def sample(self, z_curr, action_t):
        """
        Uniform sample interface matching FlowMatching dynamics.
        
        Args:
            z_curr: Current latent state tensor.
            action_t: Action tensor applied at the current step.

        Returns:
            The predicted next latent state tensor.
        """
        return self.forward(z_curr, action_t)

    def rollout(self, z_0, actions_sequence):
        """
        Predicts a multi-step trajectory of future latents autoregressively 
        given an initial latent state and an action sequence.

        Args:
            z_0: Initial latent state tensor.
            actions_sequence: Sequence of actions over time of shape [Batch, Steps, Action_Dim].

        Returns:
            A stacked tensor containing the full sequence of predicted future latents.
        """
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


class FlowMatchingDynamics(nn.Module):
    """
    Generative conditional Flow Matching transition model that models continuous vector fields 
    to map a distribution of random noise to the next latent state distribution.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, n_steps: int):
        """
        Args:
            latent_dim: Dimensionality of the latent state space.
            action_dim: Dimensionality of the action vector.
            hidden_dim: Number of hidden units in the neural network layers.
            n_steps: Number of integration steps used during sampling/inference.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.n_steps    = n_steps
        self.hidden_dim = hidden_dim

        sin_dim = min(hidden_dim, 128)  
        self.time_proj = nn.Sequential(
            nn.Linear(sin_dim, hidden_dim),
            nn.SiLU(),
        )
        self._sin_dim = sin_dim

        vnet_in = latent_dim + hidden_dim + latent_dim + action_dim
        self.v_net = nn.Sequential(
            nn.Linear(vnet_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _sinusoidal_embed(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal time embeddings for conditional continuous-time modeling.

        Args:
            timesteps: One-dimensional tensor containing integration times in the [0, 1] range.

        Returns:
            The raw Fourier embedding tensor.
        """
        half  = self._sin_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) *
            torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = (timesteps[:, None].float() * 1000.0) * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def predict_velocity(self, z_t: torch.Tensor, t: torch.Tensor,
                         z_cond: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        """
        Predicts the velocity vector field at state z_t and time step t, 
        conditioned on the current state and action.

        Args:
            z_t: Noisy latent state tensor at continuous integration time t.
            t: The continuous time step scalar or tensor.
            z_cond: Conditioning latent context (current state z_t).
            action_t: Action command applied at the current transition step.

        Returns:
            The predicted velocity vector field tensor matching the latent dimensionality.
        """
        t_emb = self._sinusoidal_embed(t)   
        t_emb = self.time_proj(t_emb)       
        x = torch.cat([z_t, t_emb, z_cond, action_t], dim=-1)
        return self.v_net(x)

    def forward(self, z_curr: torch.Tensor, action_t: torch.Tensor,
                z_next: torch.Tensor = None):
        """
        Computes either the training loss objectives or samples the model 
        if z_next is missing.

        Args:
            z_curr: Conditioning current latent state tensor.
            action_t: Action input applied at this step.
            z_next: Target destination latent state tensor (ground truth next step).

        Returns:
            If training (z_next is given), returns a tuple containing the predicted velocity field 
            and the target true velocity vector. Otherwise, samples the next state.
        """
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
        """
        Integrates the predicted vector field starting from standard Gaussian noise 
        using Euler integration to sample the next state distribution.

        Args:
            z_cond: Current latent state conditioning context.
            action_t: Action tensor applied at this transition step.

        Returns:
            The sampled next latent state tensor after full ODE integration.
        """
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
        """
        Autoregressively rolls out a multi-step trajectory of future latent states 
        by sequentially sampling through the conditional continuous-time Flow Matching ODEs.

        Args:
            z_img_0: Initial latent frame state tensor.
            actions_sequence: Sequence of action commands of shape [Batch, Steps, Action_Dim].

        Returns:
            A stacked tensor containing the trajectory of sampled future latent states.
        """
        _, N, _ = actions_sequence.shape
        pred_img_latents = []
        z_curr = z_img_0

        for step in range(N):
            action = actions_sequence[:, step, :]
            z_next = self.sample(z_curr, action)
            pred_img_latents.append(z_next)
            z_curr = z_next

        return torch.stack(pred_img_latents, dim=1)