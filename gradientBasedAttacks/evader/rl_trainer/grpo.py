"""
GRPO Reinforcement Learning Trainer
--------------------------------------
Reward = 1 - detector_AI_confidence
Advantage = (r_i - mean(r)) / std(r)  [relative, within-group baseline]
Loss = -E[A_i * log_prob(h_i | x)] + kl_beta * KL(policy || base)

No value network needed — GRPO uses relative reward as baseline.
Reference: AuthorMist (David & Gervais, 2025), DeepSeekMath (Shao et al., 2024)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def grpo_loss(
    log_probs: torch.Tensor,     # [G] — log prob of each candidate
    rewards:   torch.Tensor,     # [G] — detector reward scores
    kl_div:    torch.Tensor,     # scalar — KL from frozen base
    kl_beta:   float = 0.05,
) -> torch.Tensor:
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return -(adv * log_probs).mean() + kl_beta * kl_div


def score_candidates(
    candidates:    List[str],
    detector,
    det_tokenizer,
    device: str,
) -> torch.Tensor:
    """Query detector for each candidate. Returns reward tensor [G]."""
    rewards = []
    for cand in candidates:
        enc = det_tokenizer(
            cand, return_tensors="pt",
            truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            ai_prob = torch.softmax(detector(**enc).logits, dim=-1)[0, 1].item()
        rewards.append(1.0 - ai_prob)
    return torch.tensor(rewards, dtype=torch.float32, device=device)
