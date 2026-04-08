import functools
import os

from textattack.goal_functions import TargetedClassification
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textflint.generation.attack import Attack  # Note that here we use the Attack from textflint

from attack.methods.constraints import build_semantic_constraint
from attack.methods.search_methods import GreedyDualWIRSpan
from attack.methods.models.ppl_model import PythiaPPLModel
from attack.methods.transformations.span_seq2seq_rewrite import SpanSeq2SeqRewrite


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_recipe(target_cls):
    def init_ppl_model():
        return PythiaPPLModel(
            seq_max_len=512,
            batch_size=8,
        )

    goal_function = functools.partial(TargetedClassification, target_class=target_cls)

    min_span_len = _env_int("HMGC_PHASE2_MIN_SPAN_LEN", 3)
    max_span_len = _env_int("HMGC_PHASE2_MAX_SPAN_LEN", 6)

    constraints = [
        RepeatModification(),
        StopwordModification(),
        MaxWordsPerturbed(max_percent=0.4),
        build_semantic_constraint(
            threshold=0.75,
            compare_against_original=True,
            window_size=50,
        ),
    ]

    transformation = SpanSeq2SeqRewrite(
        rewrite_model_name=os.environ.get("HMGC_PHASE2_REWRITE_MODEL", "google/flan-t5-base"),
        num_candidate=_env_int("HMGC_PHASE2_NUM_CANDIDATES", 8),
        max_source_length=_env_int("HMGC_PHASE2_MAX_SOURCE_LENGTH", 512),
        max_new_tokens=_env_int("HMGC_PHASE2_MAX_NEW_TOKENS", 48),
        temperature=_env_float("HMGC_PHASE2_TEMPERATURE", 0.9),
        top_p=_env_float("HMGC_PHASE2_TOP_P", 0.95),
        do_sample=_env_bool("HMGC_PHASE2_DO_SAMPLE", True),
        num_beams=_env_int("HMGC_PHASE2_NUM_BEAMS", 4),
        min_span_len=min_span_len,
        max_span_len=max_span_len,
        max_rewrite_words=_env_int("HMGC_PHASE2_MAX_REWRITE_WORDS", 32),
        allow_short_span_fallback=_env_bool("HMGC_PHASE2_ALLOW_SHORT_SPAN_FALLBACK", True),
    )

    search_method = GreedyDualWIRSpan(
        alpha=_env_float("HMGC_PHASE2_ALPHA", 0.2),
        wir_method="gradient",
        min_span_len=min_span_len,
        max_span_len=max_span_len,
        max_spans_considered=_env_int("HMGC_PHASE2_MAX_SPANS_CONSIDERED", 48),
        allow_unigram_fallback=_env_bool("HMGC_PHASE2_ALLOW_UNIGRAM_FALLBACK", True),
    )
    search_method.load_llm_model = init_ppl_model

    attacking = Attack(goal_function, constraints, transformation, search_method)
    return attacking
