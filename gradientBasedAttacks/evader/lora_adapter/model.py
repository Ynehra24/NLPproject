"""
Hybrid Gradient-RL Evader — LoRA Adapter on Frozen T5-base
============================================================
Composite loss: L = λ·L_grad + (1-λ)·L_rl + α·L_sem

Train:
    python evader/lora_adapter/model.py \
      --train data/splits/train.csv \
      --val   data/splits/val.csv \
      --lam 0.5 --alpha 0.3 \
      --epochs 3 --batch-size 8 \
      --output results/evader_outputs/lambda_0.5 \
      --device cuda
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bert_score import score as bert_score
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)


# ── Dataset ─────────────────────────────────────────────────────────────────

class HC3Dataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_len: int = 256):
        df = pd.read_csv(csv_path)
        self.texts = df[df["source"] == "ai"]["text"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            "paraphrase: " + self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, self.texts[idx]


# ── Model builder ────────────────────────────────────────────────────────────

def build_model(base: str = "t5-base", r: int = 16, lora_alpha: int = 32):
    """Frozen T5-base with LoRA adapter in attention layers."""
    tokenizer = T5Tokenizer.from_pretrained(base)
    base_model = T5ForConditionalGeneration.from_pretrained(base)

    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Pseudo-embedding gradient loss ───────────────────────────────────────────

def compute_l_grad(
    token_logits: torch.Tensor,   # [B, L, vocab]
    detector,
    det_tokenizer,
    original_texts,
    device: str,
) -> torch.Tensor:
    """
    Construct pseudo-embeddings and backpropagate from surrogate detector.
    e_pseudo = Σ_v softmax(logits)_v · E_det(v)
    """
    det_embed = detector.roberta.embeddings.word_embeddings.weight  # [V_det, H]
    token_probs = F.softmax(token_logits, dim=-1)                   # [B, L, V_t5]

    # T5 vocab may differ from RoBERTa — project via shared indices up to min vocab
    min_vocab = min(token_probs.shape[-1], det_embed.shape[0])
    probs_clipped = token_probs[:, :, :min_vocab]                   # [B, L, min_V]
    pseudo = torch.matmul(probs_clipped, det_embed[:min_vocab])     # [B, L, H]

    # Detector forward on pseudo-embeddings
    logits = detector(inputs_embeds=pseudo).logits                  # [B, 2]
    human_targets = torch.zeros(logits.size(0), dtype=torch.long, device=device)
    return F.cross_entropy(logits, human_targets)


# ── GRPO reward loss ─────────────────────────────────────────────────────────

def compute_l_rl(
    model,
    tokenizer,
    detector,
    det_tokenizer,
    texts,
    device: str,
    G: int = 4,
    max_new: int = 128,
    kl_beta: float = 0.05,
) -> torch.Tensor:
    """
    Sample G paraphrases, score with detector, compute GRPO advantage loss.
    """
    all_log_probs, all_rewards = [], []

    for text in texts:
        inp = tokenizer(
            "paraphrase: " + text,
            return_tensors="pt", truncation=True, max_length=256
        ).to(device)

        group_lp, group_r = [], []
        for _ in range(G):
            out = model.generate(
                **inp, max_new_tokens=max_new,
                do_sample=True, temperature=0.9,
                output_scores=True, return_dict_in_generate=True
            )
            ids = out.sequences[0][inp.input_ids.shape[1]:]
            # Log prob of generated sequence
            lp = sum(
                F.log_softmax(s, dim=-1)[0, ids[i]].item()
                for i, s in enumerate(out.scores)
                if i < len(ids)
            )
            cand = tokenizer.decode(ids, skip_special_tokens=True)

            # Detector reward
            enc = det_tokenizer(
                cand, return_tensors="pt",
                truncation=True, max_length=256
            ).to(device)
            with torch.no_grad():
                ai_prob = torch.softmax(detector(**enc).logits, dim=-1)[0, 1].item()
            reward = 1.0 - ai_prob   # higher = more human-like

            group_lp.append(lp)
            group_r.append(reward)

        r_t  = torch.tensor(group_r,  dtype=torch.float32, device=device)
        lp_t = torch.tensor(group_lp, dtype=torch.float32, device=device)
        adv  = (r_t - r_t.mean()) / (r_t.std() + 1e-8)
        all_log_probs.append(lp_t)
        all_rewards.append(adv)

    log_probs = torch.stack(all_log_probs).mean()
    advantages = torch.stack(all_rewards).mean()
    return -(advantages * log_probs)


# ── Semantic loss ─────────────────────────────────────────────────────────────

def compute_l_sem(originals, paraphrases, device: str) -> torch.Tensor:
    """BERTScore F1 between original and paraphrase (lower = more similar)."""
    _, _, F = bert_score(paraphrases, originals, lang="en", device=device, verbose=False)
    return (1.0 - F.mean()).to(device)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = args.device
    torch.manual_seed(42)

    print("Loading models...")
    model, tokenizer = build_model()
    model = model.to(device)

    det_tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
    detector = AutoModelForSequenceClassification.from_pretrained(
        "Hello-SimpleAI/chatgpt-detector-roberta"
    ).to(device)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    dataset = HC3Dataset(args.train, tokenizer)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )

    lam, alpha = args.lam, args.alpha
    print(f"λ={lam}, α={alpha} | Training for {args.epochs} epochs")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch, texts in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass — get logits over T5 vocab
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=input_ids,
            )
            token_logits = out.logits   # [B, L, V]

            # Generate paraphrases for L_sem and L_rl
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128, do_sample=False
                )
            paraphrases = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # Compute loss components
            l_grad = compute_l_grad(token_logits, detector, det_tokenizer, texts, device)
            l_sem  = compute_l_sem(list(texts), paraphrases, device)

            if lam < 1.0:
                l_rl = compute_l_rl(
                    model, tokenizer, detector, det_tokenizer,
                    list(texts[:2]), device  # small subset per batch for speed
                )
            else:
                l_rl = torch.tensor(0.0, device=device)

            loss = lam * l_grad + (1 - lam) * l_rl + alpha * l_sem

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    # Save adapter
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    with open(f"{args.output}/config.json", "w") as f:
        json.dump({"lambda": lam, "alpha": alpha, "epochs": args.epochs}, f, indent=2)
    print(f"Saved → {args.output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",      default="data/splits/train.csv")
    p.add_argument("--val",        default="data/splits/val.csv")
    p.add_argument("--lam",        type=float, default=0.5)
    p.add_argument("--alpha",      type=float, default=0.3)
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--output",     default="results/evader_outputs/lambda_0.5")
    p.add_argument("--device",     default="cuda")
    train(p.parse_args())
