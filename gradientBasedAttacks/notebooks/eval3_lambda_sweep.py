# ============================================================
# EVAL 3: Full λ-Sweep Training + Evaluation
# Run in Google Colab (T4 GPU, ~2-3 hrs per λ value)
# Checkpoints saved to Google Drive — safe to resume after disconnect
# ============================================================

# ── CELL 1: Mount Drive + Install ───────────────────────────
from google.colab import drive
drive.mount('/content/drive')

SAVE_DIR = "/content/drive/MyDrive/NLP_gradient_evasion"
import os; os.makedirs(SAVE_DIR, exist_ok=True)

# !pip install transformers peft datasets huggingface_hub bert-score rouge-score accelerate -q


# ── CELL 2: Imports ─────────────────────────────────────────
import json, random, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as rouge_lib
from huggingface_hub import hf_hub_download
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(42)
random.seed(42)


# ── CELL 3: Load HC3 full dataset ───────────────────────────
filepath = hf_hub_download(
    repo_id="Hello-SimpleAI/HC3",
    filename="all.jsonl",
    repo_type="dataset"
)

ai_texts, human_texts = [], []
with open(filepath) as f:
    for line in f:
        item = json.loads(line)
        for ans in item.get("chatgpt_answers", []):
            if ans and len(ans.strip()) > 50:
                ai_texts.append(ans.strip())
        for ans in item.get("human_answers", []):
            if ans and len(ans.strip()) > 50:
                human_texts.append(ans.strip())

random.shuffle(ai_texts)
random.shuffle(human_texts)

# Fixed splits — same seed as baselines
test_ai,  rest_ai  = ai_texts[:1000],    ai_texts[1000:]
test_hu,  rest_hu  = human_texts[:1000], human_texts[1000:]
val_ai,   train_ai = rest_ai[:500],      rest_ai[500:]

print(f"Train AI : {len(train_ai)}")
print(f"Val AI   : {len(val_ai)}")
print(f"Test AI  : {len(test_ai)}  |  Test Human: {len(test_hu)}")


# ── CELL 4: Load frozen surrogate detector ───────────────────
DET_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
det_tok  = AutoTokenizer.from_pretrained(DET_NAME)
detector = AutoModelForSequenceClassification.from_pretrained(DET_NAME).to(device)
detector.eval()
for p in detector.parameters():
    p.requires_grad_(False)
print("RoBERTa surrogate loaded and frozen.")


# ── CELL 5: Build LoRA adapter on T5-base ───────────────────
def build_model(base="t5-base"):
    tokenizer  = T5Tokenizer.from_pretrained(base)
    base_model = T5ForConditionalGeneration.from_pretrained(base)
    lora_cfg   = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, lora_cfg).to(device)
    model.print_trainable_parameters()
    return model, tokenizer


# ── CELL 6: Loss functions ───────────────────────────────────
def compute_l_grad(token_logits):
    """Pseudo-embedding adversarial loss (white-box)."""
    det_embed = detector.roberta.embeddings.word_embeddings.weight  # [V_det, H]
    probs     = F.softmax(token_logits, dim=-1)                    # [B, L, V]
    min_v     = min(probs.shape[-1], det_embed.shape[0])
    pseudo    = torch.matmul(probs[:, :, :min_v], det_embed[:min_v])  # [B, L, H]
    logits    = detector(inputs_embeds=pseudo).logits
    targets   = torch.zeros(logits.size(0), dtype=torch.long, device=device)
    return F.cross_entropy(logits, targets)


def compute_l_rl(model, tokenizer, texts, G=4, max_new=128):
    """GRPO reward loss (black-box)."""
    loss_accum = torch.tensor(0.0, device=device)
    for text in texts:
        inp = tokenizer(
            "paraphrase: " + text,
            return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        lps, rewards = [], []
        for _ in range(G):
            out  = model.generate(
                **inp, max_new_tokens=max_new, do_sample=True, temperature=0.9,
                output_scores=True, return_dict_in_generate=True
            )
            ids  = out.sequences[0][inp.input_ids.shape[1]:]
            lp   = sum(F.log_softmax(s, dim=-1)[0, ids[i]].item()
                       for i, s in enumerate(out.scores) if i < len(ids))
            cand = tokenizer.decode(ids, skip_special_tokens=True)
            enc  = det_tok(cand, return_tensors="pt",
                           truncation=True, max_length=256).to(device)
            with torch.no_grad():
                ai_p = torch.softmax(detector(**enc).logits, dim=-1)[0, 1].item()
            lps.append(lp)
            rewards.append(1.0 - ai_p)

        r_t  = torch.tensor(rewards, device=device)
        lp_t = torch.tensor(lps,     device=device)
        adv  = (r_t - r_t.mean()) / (r_t.std() + 1e-8)
        loss_accum = loss_accum - (adv * lp_t).mean()
    return loss_accum / len(texts)


def compute_l_sem(originals, paraphrases):
    """BERTScore semantic preservation."""
    _, _, F = bert_score_fn(paraphrases, originals, lang="en",
                            device=device, verbose=False)
    return (1.0 - F.mean()).to(device)


# ── CELL 7: Evaluation helpers ───────────────────────────────
def generate_paraphrases(model, tokenizer, texts, max_new=128, batch_size=8):
    model.eval()
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc   = tokenizer(
            ["paraphrase: " + t for t in batch],
            return_tensors="pt", padding=True,
            truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        results += tokenizer.batch_decode(ids, skip_special_tokens=True)
    return results


def eval_asr(texts, batch_size=32):
    """% evaded texts classified as human by surrogate."""
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = det_tok(texts[i:i+batch_size], return_tensors="pt",
                      padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            preds += torch.argmax(detector(**enc).logits, dim=-1).cpu().tolist()
    return sum(p == 0 for p in preds) / len(preds) * 100   # label 0 = human


def eval_bertscore(originals, paraphrases):
    _, _, F = bert_score_fn(paraphrases, originals, lang="en",
                            device=device, verbose=False)
    return F.mean().item()


def eval_rouge(originals, paraphrases):
    sc = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    return np.mean([sc.score(o, p)["rougeL"].fmeasure
                    for o, p in zip(originals, paraphrases)])


# ── CELL 8: Training function (one λ) ───────────────────────
def train_one_lambda(
    lam,
    alpha       = 0.3,
    epochs      = 3,
    batch_size  = 8,
    rl_per_batch= 2,      # how many texts do RL per batch step
    max_train   = 5000,   # AI samples used for training
):
    lam_tag   = str(lam).replace(".", "_")
    save_path = f"{SAVE_DIR}/lambda_{lam_tag}"
    os.makedirs(save_path, exist_ok=True)

    # Skip if already finished
    done_file = f"{save_path}/DONE.json"
    if os.path.exists(done_file):
        print(f"λ={lam} already done — skipping.")
        with open(done_file) as f:
            return json.load(f)

    print(f"\n{'='*55}")
    print(f"  λ = {lam}   (gradient weight: {lam}   RL weight: {1-lam})")
    print(f"{'='*55}")

    model, tokenizer = build_model()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )

    train_data = train_ai[:max_train]
    epoch_loss = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        random.shuffle(train_data)

        for i in tqdm(range(0, len(train_data), batch_size),
                      desc=f"  Epoch {epoch+1}/{epochs}  λ={lam}"):
            batch = train_data[i:i+batch_size]
            enc   = tokenizer(
                ["paraphrase: " + t for t in batch],
                return_tensors="pt", padding=True,
                truncation=True, max_length=256
            ).to(device)

            out = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                decoder_input_ids=enc.input_ids,
            )

            loss = torch.tensor(0.0, device=device)

            if lam > 0:
                loss = loss + lam * compute_l_grad(out.logits)

            if lam < 1.0 and len(batch) > 0:
                loss = loss + (1 - lam) * compute_l_rl(
                    model, tokenizer, batch[:rl_per_batch]
                )

            # Semantic constraint
            with torch.no_grad():
                gen_ids = model.generate(
                    enc.input_ids, attention_mask=enc.attention_mask,
                    max_new_tokens=128, do_sample=False
                )
            paras = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            loss  = loss + alpha * compute_l_sem(batch, paras)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        n_batches  = max(1, len(train_data) // batch_size)
        epoch_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch+1} | Avg loss: {epoch_loss:.4f}")

        # Save checkpoint after every epoch → safe against disconnects
        ckpt = f"{save_path}/epoch_{epoch+1}"
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"  Checkpoint saved → {ckpt}")

    # ── Post-training evaluation ─────────────────────────────
    print(f"\n  Generating paraphrases on test set (500 samples)...")
    paras = generate_paraphrases(model, tokenizer, test_ai[:500])

    asr = eval_asr(paras)
    bs  = eval_bertscore(test_ai[:500], paras)
    rl  = eval_rouge(test_ai[:500], paras)

    # Save evaded samples CSV (compatible with team pipeline)
    pd.DataFrame({
        "id":            [f"hc3_ai_{i:05d}_evaded" for i in range(len(paras))],
        "text":          paras,
        "source":        "ai",
        "attack_type":   "gradient",
        "attack_owner":  "udaiveer",
        "generator_model": "gpt3.5-turbo",
        "original_text": test_ai[:500],
    }).to_csv(f"{save_path}/evaded.csv", index=False)

    results = {
        "lambda":       lam,
        "asr":          round(asr, 2),
        "bertscore_f1": round(bs, 4),
        "rouge_l":      round(rl, 4),
        "loss_final":   round(epoch_loss, 4),
    }

    with open(done_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  λ={lam} COMPLETE")
    print(f"  ASR: {asr:.1f}%  |  BERTScore: {bs:.4f}  |  ROUGE-L: {rl:.4f}")
    return results


# ── CELL 9: Run full sweep ───────────────────────────────────
# Trains λ = 1.0 → 0.75 → 0.5 → 0.25 → 0.0
# If session dies, re-run from Cell 1 — completed λ values are automatically skipped.

import gc

LAMBDA_VALUES = [1.0, 0.75, 0.5, 0.25, 0.0]
all_results   = []

for lam in LAMBDA_VALUES:
    res = train_one_lambda(
        lam         = lam,
        alpha       = 0.3,
        epochs      = 3,
        batch_size  = 8,
        rl_per_batch= 2,
        max_train   = 5000,
    )
    all_results.append(res)
    gc.collect()
    torch.cuda.empty_cache()


# ── CELL 10: Summary + save ──────────────────────────────────
summary = pd.DataFrame(all_results).sort_values("lambda")
summary.to_csv(f"{SAVE_DIR}/lambda_sweep_results.csv", index=False)

print("\n" + "="*60)
print("FINAL λ-SWEEP SUMMARY")
print("="*60)
print(summary.to_string(index=False))

# Baseline (no attack) for reference
baseline_det = eval_asr(test_ai[:500])
print(f"\nBaseline (no evasion): {100 - baseline_det:.1f}% detected as AI")
print(f"Best ASR achieved:     {summary['asr'].max():.1f}%  (λ={summary.loc[summary['asr'].idxmax(), 'lambda']})")
print(f"Best BERTScore:        {summary['bertscore_f1'].max():.4f}")
print(f"\nResults saved → {SAVE_DIR}/lambda_sweep_results.csv")


# ── CELL 11: Iterative loop ASR@K ───────────────────────────
# Run the best λ model through K refinement rounds

best_lam = summary.loc[summary["asr"].idxmax(), "lambda"]
best_tag  = str(best_lam).replace(".", "_")
print(f"\nRunning iterative loop with best λ={best_lam}...")

# Reload best model
best_model, best_tok = build_model()
best_model.load_adapter(f"{SAVE_DIR}/lambda_{best_tag}/epoch_3")
best_model.eval()

def iterative_asr(model, tokenizer, texts, K=5, threshold=0.5, batch_size=32):
    """
    For each text, refine up to K rounds until detector confidence < threshold.
    Returns ASR@k for each k.
    """
    current = list(texts)
    asr_at_k = {}

    for k in range(1, K+1):
        current = generate_paraphrases(model, tokenizer, current)

        # Check detector confidence
        still_ai = []
        successes = 0
        for i, text in enumerate(current):
            enc = det_tok(text, return_tensors="pt",
                          truncation=True, max_length=512).to(device)
            with torch.no_grad():
                ai_prob = torch.softmax(detector(**enc).logits, dim=-1)[0, 1].item()
            if ai_prob < threshold:
                successes += 1
                still_ai.append(texts[i])  # keep original for further rounds
            else:
                still_ai.append(text)      # keep refining

        asr_at_k[f"ASR@{k}"] = round(successes / len(texts) * 100, 2)
        current = still_ai
        print(f"  Round {k}: ASR@{k} = {asr_at_k[f'ASR@{k}']}%")

    return asr_at_k

asr_k = iterative_asr(best_model, best_tok, test_ai[:200], K=5)
print("\nIterative Inference Loop Results:")
for k, v in asr_k.items():
    print(f"  {k}: {v}%")

pd.DataFrame([asr_k]).to_csv(f"{SAVE_DIR}/asr_at_k.csv", index=False)
print(f"Saved → {SAVE_DIR}/asr_at_k.csv")
