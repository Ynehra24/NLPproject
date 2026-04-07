"""
Pseudo-Embedding Gradient Bridge
----------------------------------
Solves discrete text non-differentiability by constructing a weighted
average over the detector's embedding matrix at each token position.

e_pseudo = Σ_v softmax(logits)_v · E_det(v)

Allows full gradient flow: detector_loss → pseudo_embeds → T5_logits → adapter
Reference: GradEscape (Meng et al., USENIX 2025)
"""

import torch
import torch.nn.functional as F


def build_pseudo_embeddings(
    token_logits: torch.Tensor,       # [B, L, V_t5]
    detector_embed: torch.Tensor,     # [V_det, H]
) -> torch.Tensor:
    """Weighted sum over detector embedding matrix."""
    token_probs = F.softmax(token_logits, dim=-1)
    min_v = min(token_probs.shape[-1], detector_embed.shape[0])
    return torch.matmul(token_probs[:, :, :min_v], detector_embed[:min_v])  # [B, L, H]


def grad_adversarial_loss(
    pseudo_embeds: torch.Tensor,
    detector,
    device: str,
) -> torch.Tensor:
    """Cross-entropy pushing detector toward human label (label=0)."""
    logits = detector(inputs_embeds=pseudo_embeds).logits
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=device)
    return F.cross_entropy(logits, targets)
