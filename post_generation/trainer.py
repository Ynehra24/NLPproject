"""
trainer.py
----------
Training loop for the StyleAwareEvader.

Design decisions
================
1. The sentence-encoder embeddings of source texts are PRE-COMPUTED once
   before training starts (stored in CPU RAM) so that each training step
   does not incur the overhead of a sentence-encoder forward pass.
   This follows Meng et al.'s note that "Est({x_i}) is computed prior to
   training, as the calculation does not involve gradient propagation."

2. Mixed-precision (fp16) training is supported via torch.cuda.amp.

3. The training loop logs to both console and (optionally) tensorboard.
   Wandb integration is added as a lightweight extension hook.

4. The "repeater SFT" stage for warm-started evaders (BERT tokeniser) is
   provided as a separate train_repeater() function.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
)

from config import Config, TrainingConfig
from data_utils import build_dataloader, build_or_load_human_stats
from evader import StyleAwareEvader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence-encoder embedding pre-computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_sentence_embeddings(
    texts: List[str],
    sentence_encoder: nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Pre-compute frozen sentence-encoder embeddings for all source texts.

    Returns:
        embeddings: (N, hidden_dim) float32 tensor on CPU.
    """
    sentence_encoder.eval()
    all_embs: List[torch.Tensor] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        out = sentence_encoder(**enc)
        # Mean-pool over sequence length
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)   # (B, H)
        all_embs.append(emb.cpu())

        if (i // batch_size) % 20 == 0:
            logger.info("Pre-computed embeddings for %d / %d samples", i + len(batch), len(texts))

    return torch.cat(all_embs, dim=0)   # (N, H)


# ---------------------------------------------------------------------------
# Repeater SFT (warm-started evader initialisation)
# ---------------------------------------------------------------------------

def train_repeater(
    evader_model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    repeater_texts: List[str],
    num_steps: int = 5_000,
    batch_size: int = 16,
    lr: float = 5e-5,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    Fine-tune the warm-started encoder-decoder as a "repeater":
    the target output is a copy of the input.

    This gives the model a sensible initialisation before adversarial training,
    preventing the evader from collapsing to random outputs in early steps.

    Reference: Section 4.3, Meng et al. (2025).
    """
    logger.info("Starting repeater SFT for %d steps…", num_steps)
    evader_model.to(device).train()
    optimiser = AdamW(evader_model.parameters(), lr=lr)

    texts_cycle = repeater_texts * (num_steps * batch_size // len(repeater_texts) + 1)

    for step in range(num_steps):
        batch = texts_cycle[step * batch_size: (step + 1) * batch_size]
        enc = tokenizer(
            batch,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Labels = input_ids (copy task)
        outputs = evader_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(evader_model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad()

        if step % 10 == 0:
            logger.info("[Repeater SFT] step %d / %d  loss=%.4f", step, num_steps, loss.item())

    logger.info("Repeater SFT complete.")


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class EvaderTrainer:
    """
    Orchestrates the full training procedure for StyleAwareEvader.

    Usage
    -----
    trainer = EvaderTrainer(config, evader_model)
    trainer.train()
    """

    def __init__(
        self,
        config: Config,
        evader_model: StyleAwareEvader,
        sentence_embeddings: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.tcfg: TrainingConfig = config.training
        self.model = evader_model
        self.device = torch.device(config.device)

        # Pre-computed source embeddings (may be None if no sentence encoder)
        self.sentence_embeddings: Optional[torch.Tensor] = sentence_embeddings

        # DataLoaders
        self.train_loader = build_dataloader(
            file_path=self.tcfg.train_data_path,
            tokenizer=evader_model.tokenizer,
            max_length=config.model.max_length,
            batch_size=self.tcfg.batch_size,
            shuffle=True,
            num_workers=config.dataloader_workers,
        )
        self.eval_loader = build_dataloader(
            file_path=self.tcfg.eval_data_path,
            tokenizer=evader_model.tokenizer,
            max_length=config.model.max_length,
            batch_size=self.tcfg.batch_size,
            shuffle=False,
            num_workers=config.dataloader_workers,
        )

        # Optimiser
        self.optimiser = AdamW(
            self.model.evader.parameters(),
            lr=self.tcfg.learning_rate,
            weight_decay=self.tcfg.weight_decay,
        )

        # LR Scheduler with linear warm-up
        total_steps = len(self.train_loader) * self.tcfg.num_epochs
        warmup_steps = int(total_steps * self.tcfg.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimiser,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed-precision scaler
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if config.fp16 and torch.cuda.is_available()
            else None
        )

        # Logging
        self.global_step = 0
        self.best_eval_loss = math.inf

        Path(self.tcfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._log_buffer: List[Dict] = []

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _one_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Execute a single gradient update step (with external accumulation).

        Returns (loss, info_dict).
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        input_embeddings = None

        print(f"Step {self.global_step + 1} forward...", flush=True)
        use_fp16 = self.scaler is not None
        with torch.autocast(device_type="mps", enabled=use_fp16):
            l_total, info = self.model.training_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_embeddings=input_embeddings,
            )
            # Scale loss by accumulation steps
            grad_accum = getattr(self.tcfg, "gradient_accumulation_steps", 1)
            l_total = l_total / grad_accum

        print(f"Step {self.global_step + 1} backward...", flush=True)
        if use_fp16:
            self.scaler.scale(l_total).backward()
        else:
            l_total.backward()

        return l_total, info

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run one pass over the eval set and return aggregated metrics."""
        self.model.eval()
        total_info: Dict[str, float] = {}
        n_batches = 0

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            _, info = self.model.training_step(input_ids, attention_mask)

            for k, v in info.items():
                total_info[k] = total_info.get(k, 0.0) + v
            n_batches += 1

        avg_info = {k: v / n_batches for k, v in total_info.items()}
        self.model.train()
        return avg_info

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Full training loop across all epochs."""
        logger.info(
            "Training start | epochs=%d | steps_per_epoch=%d | "
            "total_steps=%d | device=%s",
            self.tcfg.num_epochs,
            len(self.train_loader),
            len(self.train_loader) * self.tcfg.num_epochs,
            self.device,
        )
        self.model.to(self.device)
        self.model.train()

        for epoch in range(1, self.tcfg.num_epochs + 1):
            epoch_loss = 0.0
            grad_accum = getattr(self.tcfg, "gradient_accumulation_steps", 1)
            
            # Start clean
            self.optimiser.zero_grad()

            for step, batch in enumerate(self.train_loader, 1):
                loss, info = self._one_step(batch)
                epoch_loss += loss.item() * grad_accum
                self.global_step += 1
                
                # Step optimizer based on accumulation steps
                if self.global_step % grad_accum == 0 or step == len(self.train_loader):
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimiser)
                        nn.utils.clip_grad_norm_(self.model.evader.parameters(), self.tcfg.grad_clip)
                        self.scaler.step(self.optimiser)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.evader.parameters(), self.tcfg.grad_clip)
                        self.optimiser.step()
                        
                    self.scheduler.step()
                    self.optimiser.zero_grad()
                    print(f"Step {self.global_step} optimization done!", flush=True)

                # Console logging
                if self.global_step % self.tcfg.log_every == 0:
                    self._log_step(epoch, step, info)

                # Periodic checkpoint
                if (
                    self.tcfg.save_every > 0
                    and self.global_step % self.tcfg.save_every == 0
                ):
                    self._save_checkpoint(tag=f"step_{self.global_step}")

            # --- End of epoch ---
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            eval_metrics = self.evaluate()

            logger.info(
                "[Epoch %d] avg_train_loss=%.4f | eval: %s",
                epoch,
                avg_epoch_loss,
                {k: f"{v:.4f}" for k, v in eval_metrics.items()},
            )

            # Save best checkpoint based on eval total loss
            if eval_metrics.get("l_total", math.inf) < self.best_eval_loss:
                self.best_eval_loss = eval_metrics["l_total"]
                self._save_checkpoint(tag="best")

            self._save_checkpoint(tag=f"epoch_{epoch}")

        # Flush log buffer
        self._write_log()
        logger.info("Training complete.  Best eval loss: %.4f", self.best_eval_loss)

    # ------------------------------------------------------------------
    # Logging & checkpointing helpers
    # ------------------------------------------------------------------

    def _log_step(self, epoch: int, step: int, info: Dict) -> None:
        row = {
            "epoch": epoch,
            "global_step": self.global_step,
            "lr": self.scheduler.get_last_lr()[0],
            **info,
        }
        self._log_buffer.append(row)
        self._write_log()  # Save continuously so we don't lose stats
        logger.info(
            "[E%d S%d] loss=%.4f l_adv=%.4f l_sem=%.4f l_style=%.4f "
            "hp_surr=%.3f kl_burst=%.4f",
            epoch, step,
            info.get("l_total", 0),
            info.get("l_adv", 0),
            info.get("l_sem", 0),
            info.get("l_style", 0),
            info.get("human_prob_surrogate", 0),
            info.get("kl_burstiness", 0),
        )

    def _save_checkpoint(self, tag: str) -> None:
        path = os.path.join(self.tcfg.output_dir, tag)
        self.model.save(path)
        logger.info("Checkpoint saved → %s", path)

    def _write_log(self) -> None:
        log_path = os.path.join(self.tcfg.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self._log_buffer, f, indent=2)
        logger.info("Training log written to %s", log_path)
