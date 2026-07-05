"""
model/modeling_moe.py
Sparse Mixture of Experts (MoE) layer implementation.

Replaces standard FFN layers in transformer models with a Sparse Top-2 MoE.

Key components:
  - SwiGLUExpert:   single SwiGLU-activated expert FFN
  - SparseMoELayer: Top-2 gating, capacity factor, auxiliary load-balancing loss,
                    and tracking of routing stats (probs, utilization, entropy, inactive %)
  - PhiMoEModel:    thin wrapper to load microsoft/Phi-mini-MoE-instruct and inspect config
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


# ---------------------------------------------------------------------------
# SwiGLU Expert
# ---------------------------------------------------------------------------

class SwiGLUExpert(nn.Module):
    """Single SwiGLU-activated FFN expert.

    SwiGLU: output = SiLU(W_gate * x) * (W_up * x), then W_down projection.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Sparse MoE Layer (Top-2)
# ---------------------------------------------------------------------------

class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts layer replacing a standard transformer FFN.

    Args:
        hidden_size:        Model hidden dimension.
        intermediate_size:  Inner dimension for each expert FFN.
        num_experts:        Total number of experts (default 8).
        top_k:              Number of experts activated per token (default 2).
        capacity_factor:    Controls expert token buffer size (default 1.25).
        aux_loss_weight:    Weight lambda for the load-balancing auxiliary loss (default 0.01).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_size     = hidden_size
        self.num_experts     = num_experts
        self.top_k           = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight

        # Expert FFNs
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

        # Gating network: linear -> softmax
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Tracking buffers (no grad, not persisted to checkpoint)
        self.register_buffer("_expert_token_counts", torch.zeros(num_experts), persistent=False)
        self.register_buffer("_total_tokens",        torch.tensor(0.0),        persistent=False)

    # ------------------------------------------------------------------
    # Auxiliary load-balancing loss (Switch Transformer, Fedus et al. 2022)
    # ------------------------------------------------------------------
    def _auxiliary_loss(
        self,
        routing_weights: torch.Tensor,
        topk_indices:    torch.Tensor,
    ) -> torch.Tensor:
        """L_aux = num_experts * sum_i(f_i * P_i)
        where:
          f_i = fraction of tokens dispatched to expert i (top-k hard assignment)
          P_i = mean routing probability for expert i (soft, differentiable)
        """
        num_tokens = routing_weights.size(0)

        # P_i: mean soft gate probability per expert
        P = routing_weights.mean(dim=0)                    # (num_experts,)

        # f_i: fraction of tokens assigned to expert i via top-k
        one_hot = torch.zeros_like(routing_weights)        # (T, num_experts)
        one_hot.scatter_(1, topk_indices, 1.0)
        f = one_hot.mean(dim=0)                            # (num_experts,)

        return self.num_experts * (f * P).sum()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output:   (batch, seq_len, hidden_size)
            aux_loss: scalar tensor (already scaled by aux_loss_weight)
        """
        batch, seq_len, hidden = x.shape
        T = batch * seq_len

        x_flat = x.view(T, hidden)                             # (T, H)

        # --- Gating ---
        logits          = self.gate(x_flat)                    # (T, E)
        routing_weights = F.softmax(logits, dim=-1)            # (T, E)

        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # Renormalize selected weights so they sum to 1 per token
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # --- Expert capacity ---
        expert_capacity = max(
            self.top_k,
            int(math.ceil(T / self.num_experts * self.capacity_factor)),
        )

        # --- Dispatch tokens to experts ---
        output_flat = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # Tokens that selected this expert in their top-k
            expert_mask    = (topk_indices == expert_idx).any(dim=-1)  # (T,)
            token_indices  = expert_mask.nonzero(as_tuple=True)[0]

            if token_indices.numel() == 0:
                continue

            # Enforce capacity (drop overflow tokens silently)
            if token_indices.numel() > expert_capacity:
                token_indices = token_indices[:expert_capacity]

            expert_input  = x_flat[token_indices]                        # (k, H)
            expert_output = self.experts[expert_idx](expert_input)       # (k, H)

            # Retrieve this expert's gate weight for each selected token
            # topk_indices[token_indices]: (k, top_k)
            # We find the position in top_k where the expert_idx is
            match         = (topk_indices[token_indices] == expert_idx).float()  # (k, top_k)
            gate_w        = (topk_weights[token_indices] * match).sum(dim=-1, keepdim=True)  # (k, 1)

            output_flat.index_add_(0, token_indices, gate_w * expert_output)

        output   = output_flat.view(batch, seq_len, hidden)
        aux_loss = self.aux_loss_weight * self._auxiliary_loss(routing_weights, topk_indices)

        # --- Update tracking stats (detached, no grad) ---
        with torch.no_grad():
            counts = torch.zeros(self.num_experts, device=x.device)
            for k in range(self.top_k):
                counts.scatter_add_(
                    0,
                    topk_indices[:, k],
                    torch.ones(T, device=x.device, dtype=torch.float),
                )
            self._expert_token_counts.add_(counts)
            self._total_tokens.add_(float(T * self.top_k))

        return output, aux_loss

    # ------------------------------------------------------------------
    # Routing statistics
    # ------------------------------------------------------------------
    def get_routing_stats(self) -> dict:
        """Return a dict of MoE routing statistics for logging / plotting."""
        with torch.no_grad():
            total = self._total_tokens.item()
            if total == 0:
                utilization = torch.zeros(self.num_experts)
            else:
                utilization = (self._expert_token_counts / total).cpu()

            probs   = utilization / (utilization.sum() + 1e-9)
            entropy = float(-(probs * (probs + 1e-9).log()).sum())

            num_inactive = int((self._expert_token_counts == 0).sum().item())
            pct_inactive = 100.0 * num_inactive / self.num_experts

        return {
            "expert_utilization":    utilization.numpy(),
            "routing_frequency":     utilization.numpy(),
            "gate_entropy":          entropy,
            "pct_inactive_experts":  pct_inactive,
            "total_tokens_routed":   total,
        }

    def reset_tracking(self):
        """Reset tracking buffers — call at the start of each evaluation epoch."""
        self._expert_token_counts.zero_()
        self._total_tokens.zero_()


# ---------------------------------------------------------------------------
# Phi-mini-MoE model wrapper
# ---------------------------------------------------------------------------

class PhiMoEModel(nn.Module):
    """Thin wrapper around microsoft/Phi-mini-MoE-instruct.

    Loads the model and exposes its configuration for downstream use
    (hidden_size, num_layers, vocab_size, etc.).
    """

    MODEL_NAME = "microsoft/Phi-mini-MoE-instruct"

    def __init__(self, model_name: Optional[str] = None, torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        name = model_name or self.MODEL_NAME
        print(f"[PhiMoEModel] Loading config from '{name}' ...")
        self.config = AutoConfig.from_pretrained(name, trust_remote_code=True)

        print(f"[PhiMoEModel] Loading model weights (dtype={torch_dtype}) ...")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        self.hidden_size = self.config.hidden_size
        self.num_layers  = self.config.num_hidden_layers
        self.vocab_size  = self.config.vocab_size

        print(
            f"[PhiMoEModel] Loaded.  hidden_size={self.hidden_size} | "
            f"num_layers={self.num_layers} | vocab_size={self.vocab_size}"
        )

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        return self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Quick smoke test (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("  SparseMoELayer — smoke test (CPU)")
    print("=" * 50)

    layer = SparseMoELayer(
        hidden_size=256, intermediate_size=512, num_experts=4, top_k=2
    )
    x_in = torch.randn(2, 8, 256)
    out, loss = layer(x_in)

    assert out.shape == x_in.shape, f"Shape mismatch: {out.shape} vs {x_in.shape}"
    print(f"  Input  shape : {x_in.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Aux loss     : {loss.item():.6f}")
    stats = layer.get_routing_stats()
    print(f"  Expert util  : {stats['expert_utilization'].round(3)}")
    print(f"  Gate entropy : {stats['gate_entropy']:.4f}")
    print(f"  Inactive %   : {stats['pct_inactive_experts']:.1f}%")
    print("  PASSED")

    if "--load-phi" in sys.argv:
        print("\n" + "=" * 50)
        print("  PhiMoEModel — load test")
        print("=" * 50)
        phi = PhiMoEModel()
        print(phi.config)
