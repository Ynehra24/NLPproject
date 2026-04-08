"""
Span-level seq2seq rewriting transformation
==========================================
"""

import os
import re

import torch
from textattack.transformations import Transformation
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class SpanSeq2SeqRewrite(Transformation):
    """Rewrites a contiguous span with a seq2seq model.

    Given a selected word span, this transformation prompts an instruction-tuned
    seq2seq model to produce alternative phrasings that preserve meaning while
    allowing structural variation.
    """

    def __init__(
        self,
        rewrite_model_name="google/flan-t5-base",
        num_candidate=8,
        max_source_length=512,
        max_new_tokens=48,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        num_beams=4,
        min_span_len=3,
        max_span_len=6,
        max_rewrite_words=32,
        allow_short_span_fallback=True,
        device=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.rewrite_model_name = rewrite_model_name
        self.num_candidate = num_candidate
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len
        self.max_rewrite_words = max_rewrite_words
        self.allow_short_span_fallback = allow_short_span_fallback

        requested_device = device or os.environ.get("TA_DEVICE")
        if requested_device and requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = None
        self._model = None

    @property
    def deterministic(self):
        return False

    def _lazy_load_model(self):
        if self._model is not None and self._tokenizer is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.rewrite_model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.rewrite_model_name)
        self._model.to(self.device)
        self._model.eval()

    def _pick_span(self, indices_to_modify):
        if not indices_to_modify:
            return None

        indices = sorted(indices_to_modify)
        best_start = indices[0]
        best_end = indices[0]

        cur_start = indices[0]
        cur_end = indices[0]
        for idx in indices[1:]:
            if idx == cur_end + 1:
                cur_end = idx
            else:
                if (cur_end - cur_start) > (best_end - best_start):
                    best_start, best_end = cur_start, cur_end
                cur_start = idx
                cur_end = idx

        if (cur_end - cur_start) > (best_end - best_start):
            best_start, best_end = cur_start, cur_end

        span_len = best_end - best_start + 1
        if span_len > self.max_span_len:
            best_end = best_start + self.max_span_len - 1

        return best_start, best_end

    @staticmethod
    def _sanitize_generated_span(text):
        text = text.strip()
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)

        prefixes = [
            "rewritten span:",
            "rewrite:",
            "answer:",
            "output:",
            "rephrased span:",
        ]
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix) :].strip()
                break

        text = text.strip(" \t\r`\"'")
        return text

    def _valid_candidate(self, candidate, source_span):
        if not candidate:
            return False

        if candidate.lower() == source_span.lower():
            return False

        num_words = len(candidate.split())
        if num_words == 0 or num_words > self.max_rewrite_words:
            return False

        return True

    @staticmethod
    def _replace_span(current_text, start_idx, end_idx, rewritten_span):
        words = current_text.words[:]
        words[start_idx] = rewritten_span
        for idx in range(start_idx + 1, end_idx + 1):
            words[idx] = ""
        return current_text.generate_new_attacked_text(words)

    def _build_prompt(self, prefix, source_span, suffix):
        prefix_text = prefix if prefix else "<EMPTY>"
        suffix_text = suffix if suffix else "<EMPTY>"

        return (
            "Rewrite the TARGET SPAN while preserving meaning and fitting the context. "
            "Prefer structural changes (clause/voice/order) over trivial synonym swaps.\n"
            f"Context before: {prefix_text}\n"
            f"Target span: {source_span}\n"
            f"Context after: {suffix_text}\n"
            "Return only the rewritten span."
        )

    def _generate_rewrites(self, prompt):
        self._lazy_load_model()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.do_sample:
            generation_kwargs = {
                "do_sample": True,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "num_beams": 1,
                "num_return_sequences": self.num_candidate,
            }
        else:
            num_beams = max(self.num_beams, self.num_candidate)
            generation_kwargs = {
                "do_sample": False,
                "num_beams": num_beams,
                "num_return_sequences": min(self.num_candidate, num_beams),
            }

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=1.1,
                **generation_kwargs,
            )

        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    def _get_transformations(self, current_text, indices_to_modify):
        if not indices_to_modify:
            return []

        span = self._pick_span(indices_to_modify)
        if span is None:
            return []

        start_idx, end_idx = span
        span_len = end_idx - start_idx + 1
        if span_len < self.min_span_len and not self.allow_short_span_fallback:
            return []

        source_span = " ".join(current_text.words[start_idx : end_idx + 1]).strip()
        if not source_span:
            return []

        prefix = " ".join(current_text.words[:start_idx]).strip()
        suffix = " ".join(current_text.words[end_idx + 1 :]).strip()
        prompt = self._build_prompt(prefix, source_span, suffix)

        rewrites = self._generate_rewrites(prompt)

        transformed_texts = []
        seen_texts = set()
        for rewrite in rewrites:
            candidate_span = self._sanitize_generated_span(rewrite)
            if not self._valid_candidate(candidate_span, source_span):
                continue

            attacked_text = self._replace_span(
                current_text,
                start_idx,
                end_idx,
                candidate_span,
            )
            if attacked_text.text == current_text.text:
                continue
            if attacked_text.text in seen_texts:
                continue
            seen_texts.add(attacked_text.text)
            transformed_texts.append(attacked_text)

            if len(transformed_texts) >= self.num_candidate:
                break

        return transformed_texts

    def extra_repr_keys(self):
        return [
            "rewrite_model_name",
            "num_candidate",
            "min_span_len",
            "max_span_len",
            "max_new_tokens",
            "do_sample",
        ]
