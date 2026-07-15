import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float,
                 decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self._embedding_dim  = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay   = decay
        self._epsilon = epsilon

        # Codebook is a buffer, not a Parameter: updated in-place via EMA, not optimizer
        embedding = torch.zeros(num_embeddings, embedding_dim)
        embedding.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer('_embedding',        embedding)
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w',            embedding.clone())
        self.register_buffer('_usage_count',      torch.zeros(num_embeddings))

    def forward(self, inputs):
        # inputs: [B, C, H, W]  →  BHWC for distance computation
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input  = inputs.view(-1, self._embedding_dim)   # [B*H*W, C]

        # Nearest-codebook-vector lookup
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # FIX (Bug 2): quantize with OLD weights BEFORE any update
        quantized = torch.matmul(encodings, self._embedding).view(input_shape)

        if self.training:
            # ── EMA codebook update ────────────────────────────────────────────
            current_counts = torch.sum(encodings, dim=0)                    # [K]
            self._ema_cluster_size.mul_(self._decay).add_(current_counts, alpha=1 - self._decay)

            # Laplace-smoothed cluster size (avoids div-by-zero on unused codes)
            n = self._ema_cluster_size.sum()
            cluster_size_smoothed = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n
            )

            dw = torch.matmul(encodings.t(), flat_input.detach())           # [K, C]
            self._ema_w.mul_(self._decay).add_(dw, alpha=1 - self._decay)

            self._embedding.data.copy_(self._ema_w / cluster_size_smoothed.unsqueeze(1))

            # ── Dead-code revival ──────────────────────────────────────────────
            # Use _ema_cluster_size (not a separate counter) as the aliveness signal
            dead_threshold = 1.0   # codes with EMA-smoothed count < 1 are considered dead
            dead_indices = torch.nonzero(self._ema_cluster_size < dead_threshold).squeeze(1)

            if len(dead_indices) > 0:
                random_idx = torch.randint(0, flat_input.size(0), (len(dead_indices),), device=inputs.device)
                sampled    = flat_input[random_idx].detach()

                reset_count = dead_threshold * 100.0   # 100.0
                self._embedding.data[dead_indices]        = sampled
                # FIX: _ema_w must be scaled by reset_count so that
                # embedding = ema_w / ema_cluster_size = (sampled*100) / 100 = sampled
                # Without this, the ratio was (sampled) / 100 → scale collapses to ~0
                self._ema_w.data[dead_indices]            = sampled * reset_count
                self._ema_cluster_size.data[dead_indices] = reset_count

        # Loss: only commitment term (codebook is updated by EMA, not gradients)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity: effective codebook usage. Range [1, num_embeddings].
        avg_probs   = encodings.mean(dim=0)
        perplexity  = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # BHWC → BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity
