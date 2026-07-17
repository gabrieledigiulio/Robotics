import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module using Exponential Moving Average (EMA) for codebook updates.
    This approach updates the embeddings directly rather than relying on optimizer gradients.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float,
                 decay: float = 0.99, epsilon: float = 1e-5):
        """
        Args:
            num_embeddings: Number of discrete embeddings in the codebook.
            embedding_dim: Dimensionality of each embedding vector.
            commitment_cost: Weighting factor for the commitment loss.
            decay: Decay factor for the EMA update.
            epsilon: Small constant for Laplace smoothing to prevent division by zero.
        """
        super().__init__()
        self._embedding_dim  = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay   = decay
        self._epsilon = epsilon

        embedding = torch.zeros(num_embeddings, embedding_dim)
        embedding.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer('_embedding',        embedding)
        self.register_buffer('_ema_cluster_size', torch.ones(num_embeddings))
        self.register_buffer('_ema_w',            embedding.clone())
        self.register_buffer('_usage_count',      torch.zeros(num_embeddings))

    def forward(self, inputs):
        """
        Args:
            inputs: Input continuous tensor of shape [B, C, H, W].

        Returns:
            A tuple containing the quantized tensor, the commitment loss, 
            and the perplexity metric.
        """
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input  = inputs.view(-1, self._embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding).view(input_shape)

        if self.training:
            current_counts = torch.sum(encodings, dim=0)
            self._ema_cluster_size.mul_(self._decay).add_(current_counts, alpha=1 - self._decay)

            cluster_size_smoothed = self._ema_cluster_size + self._epsilon

            dw = torch.matmul(encodings.t(), flat_input.detach())
            self._ema_w.mul_(self._decay).add_(dw, alpha=1 - self._decay)

            self._embedding.data.copy_(self._ema_w / cluster_size_smoothed.unsqueeze(1))

            dead_threshold = 1.0
            dead_indices = torch.nonzero(self._ema_cluster_size < dead_threshold).squeeze(1)

            if len(dead_indices) > 0:
                random_idx = torch.randint(0, flat_input.size(0), (len(dead_indices),), device=inputs.device)
                sampled    = flat_input[random_idx].detach()

                avg_cluster_size = self._ema_cluster_size.mean().item()
                reset_count = max(1.0, avg_cluster_size) 
                
                self._embedding.data[dead_indices]        = sampled
                self._ema_w.data[dead_indices]            = sampled * reset_count
                self._ema_cluster_size.data[dead_indices] = reset_count

        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction="none").sum(dim=[1, 2, 3]).mean()
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs   = encodings.mean(dim=0)
        perplexity  = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity