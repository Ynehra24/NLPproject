"""
trainer.py
----------
Two-Phase Training loop for the StyleAwareEvader.

TRAINING STRATEGY
=================
Phase 1 — Warm-Up (Reconstruction/Paraphrase SFT):
    Standard seq2seq training with labels=input_ids (cross-entropy).
    The model learns to COPY the input faithfully and generate coherent text.
    This prevents the catastrophic collapse seen when adversarial loss is
    applied to an untrained model.

Phase 2 — Adversarial Fine-Tuning:
    Gentle adversarial training with the joint loss (L_adv + L_sem + L_style + L_fluency).
    Uses a much lower learning rate to avoid corrupting the paraphrase ability
    established in Phase 1. Also blends reconstruction loss to anchor outputs.

Design decisions
================
1. Two separate optimizers: Phase 1 uses standard LR, Phase 2 uses much lower LR.
2. Phase 1 uses BART's native labels= interface for correct teacher-forcing.
3. Phase 2 adds adversarial gradient gently on top of the reconstruction baseline.
4. Checkpoints are saved after each phase.
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
# Main Trainer (Two-Phase)
# ---------------------------------------------------------------------------

class EvaderTrainer:
    """
    Two-Phase trainer for StyleAwareEvader.

    Phase 1: Warm-up — standard seq2seq copy training.
    Phase 2: Adversarial — gentle adversarial fine-tuning.

    Usage
    -----
    trainer = EvaderTrainer(config, evader_model)
    trainer.train()
    """

    def __init__(
        self,
        config: Config,
        evader_model: StyleAwareEvader,
    ):
        self.config = config
        self.tcfg: TrainingConfig = config.training
        self.model = evader_model
        self.device = torch.device(config.device)

        # DataLoaders
        self.train_loader = build_dataloader(
            file_path=self.tcfg.train_data_path,
            tokenizer=evader_model.tokenizer,
            max_length=config.model.max_length,
            batch_size=self.tcfg.batch_size,
            shuffle=True,
            num_workers=config.dataloader_workers,
            max_samples=self.tcfg.max_train_samples,
        )
        self.eval_loader = build_dataloader(
            file_path=self.tcfg.eval_data_path,
            tokenizer=evader_model.tokenizer,
            max_length=config.model.max_length,
            batch_size=self.tcfg.batch_size,
            shuffle=False,
            num_workers=config.dataloader_workers,
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
    # Phase 1: Warm-Up Training (Reconstruction/Paraphrase SFT)
    # ------------------------------------------------------------------

    def _run_warmup_phase(self) -> None:
        """
        Phase 1: Train the evader as a standard seq2seq copier/paraphraser.
        
        Uses standard cross-entropy loss with labels=input_ids.
        This establishes coherent text generation before adversarial training.
        """
        warmup_epochs = self.tcfg.warmup_epochs
        if warmup_epochs <= 0:
            logger.info("Skipping Phase 1 (warmup_epochs=0)")
            return

        logger.info(
            "=" * 60 + "\n"
            "PHASE 1: WARM-UP (Reconstruction/Paraphrase SFT)\n"
            "Epochs: %d | Steps/epoch: %d | LR: %.1e\n" +
            "=" * 60,
            warmup_epochs, len(self.train_loader), self.tcfg.warmup_learning_rate,
        )

        # Separate optimizer for Phase 1
        optimizer = AdamW(
            self.model.evader.parameters(),
            lr=self.tcfg.warmup_learning_rate,
            weight_decay=self.tcfg.weight_decay,
        )
        total_steps = len(self.train_loader) * warmup_epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.model.to(self.device)
        self.model.train()
        grad_accum = self.tcfg.gradient_accumulation_steps

        for epoch in range(1, warmup_epochs + 1):
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                loss, info = self.model.reconstruction_step(input_ids, attention_mask)
                loss = loss / grad_accum
                loss.backward()

                epoch_loss += loss.item() * grad_accum
                self.global_step += 1

                if self.global_step % grad_accum == 0 or step == len(self.train_loader):
                    nn.utils.clip_grad_norm_(self.model.evader.parameters(), self.tcfg.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if self.global_step % self.tcfg.log_every == 0:
                    logger.info(
                        "[Phase1 E%d S%d] recon_loss=%.4f lr=%.2e",
                        epoch, step, info.get("l_reconstruction", 0),
                        scheduler.get_last_lr()[0],
                    )
                    self._log_buffer.append({
                        "phase": "warmup",
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "lr": scheduler.get_last_lr()[0],
                        **info,
                    })

            avg_loss = epoch_loss / len(self.train_loader)
            logger.info(
                "[Phase1 Epoch %d] avg_loss=%.4f", epoch, avg_loss,
            )

            # Save Phase 1 checkpoint
            self._save_checkpoint(tag=f"warmup_epoch_{epoch}")

        # Save final warm-up checkpoint
        self._save_checkpoint(tag="warmup_done")
        self._write_log()
        logger.info("Phase 1 (Warm-Up) complete.")

    # ------------------------------------------------------------------
    # Phase 2: Adversarial Fine-Tuning
    # ------------------------------------------------------------------

    def _run_adversarial_phase(self) -> None:
        """
        Phase 2: Gentle adversarial fine-tuning.
        
        Uses the joint loss (adversarial + semantic + stylometric + fluency)
        with a much lower learning rate to avoid corrupting the paraphrase
        ability from Phase 1.
        """
        num_epochs = self.tcfg.num_epochs
        if num_epochs <= 0:
            logger.info("Skipping Phase 2 (num_epochs=0)")
            return

        logger.info(
            "\n" + "=" * 60 + "\n"
            "PHASE 2: ADVERSARIAL FINE-TUNING\n"
            "Epochs: %d | Steps/epoch: %d | LR: %.1e\n" +
            "=" * 60,
            num_epochs, len(self.train_loader), self.tcfg.learning_rate,
        )

        # Fresh optimizer with lower LR for Phase 2
        optimizer = AdamW(
            self.model.evader.parameters(),
            lr=self.tcfg.learning_rate,
            weight_decay=self.tcfg.weight_decay,
        )
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * self.tcfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.model.to(self.device)
        self.model.train()
        grad_accum = self.tcfg.gradient_accumulation_steps

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                loss, info = self.model.training_step(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = loss / grad_accum
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * grad_accum
                self.global_step += 1

                if self.global_step % grad_accum == 0 or step == len(self.train_loader):
                    if self.scaler is not None:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.evader.parameters(), self.tcfg.grad_clip)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.evader.parameters(), self.tcfg.grad_clip)
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()

                # Console logging
                if self.global_step % self.tcfg.log_every == 0:
                    self._log_step(epoch, step, info, scheduler)

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
                "[Phase2 Epoch %d] avg_train_loss=%.4f | eval: %s",
                epoch,
                avg_epoch_loss,
                {k: f"{v:.4f}" for k, v in eval_metrics.items()
                 if isinstance(v, (int, float)) and k != "phase"},
            )

            # Save best checkpoint based on eval total loss
            if eval_metrics.get("l_total", math.inf) < self.best_eval_loss:
                self.best_eval_loss = eval_metrics["l_total"]
                self._save_checkpoint(tag="best")

            self._save_checkpoint(tag=f"epoch_{epoch}")

        self._write_log()
        logger.info("Phase 2 (Adversarial) complete. Best eval loss: %.4f", self.best_eval_loss)

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
                if isinstance(v, (int, float)):
                    total_info[k] = total_info.get(k, 0.0) + v
            n_batches += 1

        avg_info = {k: v / n_batches for k, v in total_info.items()}
        self.model.train()
        return avg_info

    # ------------------------------------------------------------------
    # Main training loop (orchestrates both phases)
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Full two-phase training: warm-up then adversarial."""
        logger.info(
            "Starting Two-Phase Training\n"
            "  Phase 1 (Warm-Up): %d epochs\n"
            "  Phase 2 (Adversarial): %d epochs\n"
            "  Batch size: %d | Grad accum: %d | Device: %s",
            self.tcfg.warmup_epochs,
            self.tcfg.num_epochs,
            self.tcfg.batch_size,
            self.tcfg.gradient_accumulation_steps,
            self.device,
        )

        # Phase 1: Warm-Up
        self._run_warmup_phase()

        # Phase 2: Adversarial Fine-Tuning
        self._run_adversarial_phase()

        # Flush log buffer
        self._write_log()
        logger.info("Training complete. Best eval loss: %.4f", self.best_eval_loss)

    # ------------------------------------------------------------------
    # Logging & checkpointing helpers
    # ------------------------------------------------------------------

    def _log_step(self, epoch: int, step: int, info: Dict, scheduler) -> None:
        row = {
            "phase": "adversarial",
            "epoch": epoch,
            "global_step": self.global_step,
            "lr": scheduler.get_last_lr()[0],
            **{k: v for k, v in info.items() if isinstance(v, (int, float))},
        }
        self._log_buffer.append(row)
        self._write_log()
        logger.info(
            "[E%d S%d] loss=%.4f l_adv=%.4f l_sem=%.4f l_style=%.4f l_recon=%.4f "
            "hp_surr=%.3f",
            epoch, step,
            info.get("l_total", 0),
            info.get("l_adv", 0),
            info.get("l_sem", 0),
            info.get("l_style", 0),
            info.get("l_reconstruction", 0),
            info.get("human_prob_surrogate", 0),
        )

    def _save_checkpoint(self, tag: str) -> None:
        path = os.path.join(self.tcfg.output_dir, tag)
        self.model.save(path)
        logger.info("Checkpoint saved → %s", path)

    def _write_log(self) -> None:
        log_path = os.path.join(self.tcfg.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(self._log_buffer, f, indent=2)
