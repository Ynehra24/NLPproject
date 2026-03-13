"""Train a RoBERTa-based detector on human vs AI text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from detectors.common.io_utils import load_dataset
from detectors.common.metrics import encode_source_labels, find_best_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RoBERTa detector")
    parser.add_argument("--train", required=True, help="Train split path (.csv/.jsonl)")
    parser.add_argument("--val", required=True, help="Validation split path (.csv/.jsonl)")
    parser.add_argument("--model-name", default="roberta-base", help="HF checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()


def prepare_dataset(df: pd.DataFrame, tokenizer, max_length: int = 512) -> Dataset:
    labels = encode_source_labels(df["source"])
    ds = Dataset.from_dict({"text": df["text"].astype(str).tolist(), "labels": labels.tolist()})

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds.map(tok, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
    preds = (probs >= 0.5).astype(int)

    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_ai": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, probs))
    else:
        out["auroc"] = 0.0
    return out


def main() -> None:
    args = parse_args()

    train_df = load_dataset(args.train)
    val_df = load_dataset(args.val)

    if "source" not in train_df.columns or train_df["source"].isna().any():
        raise ValueError("Train data must contain non-null 'source' labels")
    if "source" not in val_df.columns or val_df["source"].isna().any():
        raise ValueError("Validation data must contain non-null 'source' labels")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_ds = prepare_dataset(train_df, tokenizer)
    val_ds = prepare_dataset(val_df, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    val_logits = trainer.predict(val_ds).predictions
    val_scores = 1.0 / (1.0 + np.exp(-(val_logits[:, 1] - val_logits[:, 0])))
    y_true = encode_source_labels(val_df["source"])
    threshold = find_best_threshold(y_true, val_scores)

    metrics = trainer.evaluate(val_ds)
    metrics["best_threshold"] = float(threshold)

    metrics_path = Path(args.output_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model and metrics saved to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
