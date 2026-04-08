"""
Greedy span-level rewrite search with dual importance ranking
=============================================================
"""

import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod


class GreedyDualWIRSpan(SearchMethod):
    """Greedy search over contiguous n-gram spans.

    Span importance combines victim-model importance and language-model
    perplexity sensitivity, then rewrites high-priority spans first.
    """

    def __init__(
        self,
        alpha=0.2,
        wir_method="gradient",
        min_span_len=3,
        max_span_len=6,
        max_spans_considered=48,
        allow_unigram_fallback=True,
    ):
        self.alpha = alpha
        self.wir_method = wir_method
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len
        self.max_spans_considered = max_spans_considered
        self.allow_unigram_fallback = allow_unigram_fallback

        self.llm_model = None

    @staticmethod
    def _delete_span(attacked_text, start_idx, end_idx):
        words = attacked_text.words[:]
        words[start_idx] = ""
        for idx in range(start_idx + 1, end_idx + 1):
            words[idx] = ""
        return attacked_text.generate_new_attacked_text(words)

    def _build_span_candidates(self, attacked_text):
        _, modifiable_indices = self.get_indices_to_order(attacked_text)
        modifiable_set = set(modifiable_indices)

        num_words = attacked_text.num_words
        if num_words == 0:
            return [], modifiable_indices

        max_span_len = min(self.max_span_len, num_words)
        min_span_len = min(self.min_span_len, max_span_len)

        spans = []
        for span_len in range(min_span_len, max_span_len + 1):
            for start_idx in range(0, num_words - span_len + 1):
                end_idx = start_idx + span_len - 1
                if all(idx in modifiable_set for idx in range(start_idx, end_idx + 1)):
                    spans.append((start_idx, end_idx))

        if not spans and self.allow_unigram_fallback:
            spans = [(idx, idx) for idx in modifiable_indices]

        return spans, modifiable_indices

    def _get_victim_importance(self, attacked_text, spans, modifiable_indices):
        if not spans:
            return np.array([]), True

        if self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            grad_output = victim_model.get_grad(attacked_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = attacked_text.align_with_model_tokens(victim_model)

            word_scores = np.zeros(attacked_text.num_words)
            for idx in modifiable_indices:
                matched_tokens = word2token_mapping[idx]
                if not matched_tokens:
                    word_scores[idx] = 0.0
                else:
                    agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    word_scores[idx] = np.linalg.norm(agg_grad, ord=1)

            span_scores = np.array(
                [np.mean(word_scores[start_idx : end_idx + 1]) for start_idx, end_idx in spans]
            )
            return min_max_normalize(span_scores), False

        leave_one_texts = [
            self._delete_span(attacked_text, start_idx, end_idx)
            for start_idx, end_idx in spans
        ]
        leave_one_results, search_over = self.get_goal_results(leave_one_texts)

        span_scores = np.zeros(len(spans))
        for idx, result in enumerate(leave_one_results):
            span_scores[idx] = result.score

        return min_max_normalize(span_scores), search_over

    def _get_llm_ppl_importance(self, attacked_text, spans):
        if not spans:
            return np.array([])

        if not self.llm_model:
            self.llm_model = self.load_llm_model()

        leave_one_texts = [attacked_text.tokenizer_input]
        leave_one_texts += [
            self._delete_span(attacked_text, start_idx, end_idx).tokenizer_input
            for start_idx, end_idx in spans
        ]

        leave_one_results = self.llm_model.eval_ppl(leave_one_texts)
        origin_ppl = leave_one_results[0]

        scores = np.array([ppl - origin_ppl for ppl in leave_one_results[1:]])
        if len(scores) < len(spans):
            scores = np.pad(scores, (0, len(spans) - len(scores)), mode="constant")

        return min_max_normalize(scores)

    def _get_span_order(self, attacked_text):
        spans, modifiable_indices = self._build_span_candidates(attacked_text)
        if not spans:
            return [], True

        victim_scores, search_over = self._get_victim_importance(
            attacked_text,
            spans,
            modifiable_indices,
        )
        llm_scores = self._get_llm_ppl_importance(attacked_text, spans)

        if len(victim_scores) == 0:
            return [], search_over

        combined_scores = (1 - self.alpha) * victim_scores + self.alpha * llm_scores
        ranked_indices = np.argsort(-combined_scores)

        if self.max_spans_considered and len(ranked_indices) > self.max_spans_considered:
            ranked_indices = ranked_indices[: self.max_spans_considered]

        span_order = [spans[idx] for idx in ranked_indices]
        return span_order, search_over

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        span_order, search_over = self._get_span_order(attacked_text)

        cur_result = initial_result
        for start_idx, end_idx in span_order:
            if search_over:
                break

            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=list(range(start_idx, end_idx + 1)),
            )
            if not transformed_text_candidates:
                continue

            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            if results[0].score <= cur_result.score:
                continue

            cur_result = results[0]
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    similarity_score = candidate.attack_attrs.get("similarity_score")
                    if similarity_score is None:
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        return cur_result

    def check_transformation_compatibility(self, transformation):
        return True

    @property
    def is_black_box(self):
        return self.wir_method != "gradient"

    def extra_repr_keys(self):
        return [
            "wir_method",
            "alpha",
            "min_span_len",
            "max_span_len",
            "max_spans_considered",
        ]


def min_max_normalize(scores):
    scores_np = np.array(scores)
    if scores_np.size == 0:
        return scores_np

    min_score = np.min(scores_np)
    max_score = np.max(scores_np)

    if np.isclose(max_score - min_score, 0):
        return np.full_like(scores_np, 1 / len(scores_np), dtype=float)
    return (scores_np - min_score) / (max_score - min_score)
