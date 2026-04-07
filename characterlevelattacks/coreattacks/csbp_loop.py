import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from composite_scorer import composite_score
from homoglyph_attack import attack_text, apply_homoglyph, apply_diacritic, is_eligible

random.seed(42)
np.random.seed(42)
@dataclass(order=True)
class Beam:
    """A single candidate in the beam."""
    score    : float          # composite score S (higher = better)
    text     : str = field(compare=False)
    round_no : int = field(compare=False, default=0)
    history  : List[str] = field(compare=False, default_factory=list)

    def __post_init__(self):
        self.history = self.history + [self.text]
ATTACK_MODES = ['homoglyph', 'diacritic', 'mixed']

def generate_candidates(text: str, n_candidates: int = 10) -> List[str]:
    """
    Generate n_candidates perturbations of text by:
    - varying attack mode (homoglyph / diacritic / mixed)
    - varying perturbation rate (0.2 – 0.6)
    - varying which tokens are targeted (random subsets)
    """
    candidates = set()
    tokens = text.split()
    eligible = [i for i, t in enumerate(tokens) if is_eligible(t)]

    if not eligible:
        return [text]

    for _ in range(n_candidates * 3):   # oversample, deduplicate
        mode    = random.choice(ATTACK_MODES)
        rate    = random.uniform(0.2, 0.6)
        # randomly mask out some eligible positions this round
        n_active = max(1, int(len(eligible) * random.uniform(0.4, 1.0)))
        active   = set(random.sample(eligible, n_active))

        new_tokens = []
        for i, tok in enumerate(tokens):
            if i in active:
                if mode == 'homoglyph':
                    new_tokens.append(apply_homoglyph(tok, rate))
                elif mode == 'diacritic':
                    new_tokens.append(apply_diacritic(tok, rate))
                else:
                    fn = random.choice([apply_homoglyph, apply_diacritic])
                    new_tokens.append(fn(tok, rate))
            else:
                new_tokens.append(tok)

        candidates.add(' '.join(new_tokens))
        if len(candidates) >= n_candidates:
            break

    return list(candidates)[:n_candidates]
def misclassifies(
    text         : str,
    classifier_fn: Callable[[str], str],
    original_label: str
) -> bool:
    """
    Returns True if the classifier now predicts a DIFFERENT label
    than the original — i.e., the attack succeeded.
    """
    return classifier_fn(text) != original_label
def csbp_loop(
    original_text  : str,
    original_label : str,
    classifier_fn  : Callable[[str], str],   # fn(text) -> label str
    K              : int   = 5,              # max rounds
    beam_width     : int   = 5,              # B: beams kept per round
    n_candidates   : int   = 10,             # candidates per beam per round
    score_weights  : Optional[dict] = None,  # override composite_score weights
    verbose        : bool  = True,
) -> dict:
    """
    CSBP (Character-level Search-Based Perturbation) beam search loop.

    At each round:
      1. Expand every beam into n_candidates perturbations
      2. Score all candidates with composite scorer S
      3. Check each for misclassification → stop early if found
      4. Prune to top-B beams by score
      5. Repeat up to K rounds

    Returns dict with:
      - 'success'        : bool
      - 'best_text'      : str
      - 'best_score'     : float
      - 'round_found'    : int or None
      - 'score_breakdown': dict of final score terms
      - 'beam_history'   : scores per round
    """
    weights       = score_weights or {}
    beam_history  = []

    # Initialise beam with original text
    init_score = composite_score(original_text, original_text, **weights)['S']
    beams      = [Beam(score=init_score, text=original_text, round_no=0)]

    best_attack     = None
    best_score      = -1.0
    round_found     = None

    for k in range(1, K + 1):
        if verbose:
            print(f"\n[CSBP] Round {k}/{K}  |  active beams: {len(beams)}")

        round_candidates: List[Beam] = []

        for beam in beams:
            candidates = generate_candidates(beam.text, n_candidates)

            for cand_text in candidates:
                # Score against the ORIGINAL (not the beam) to avoid drift
                result = composite_score(original_text, cand_text, **weights)
                S      = result['S']

                # Misclassification check
                if misclassifies(cand_text, classifier_fn, original_label):
                    if verbose:
                        print(f"  ✓ Attack succeeded at round {k}  S={S:.4f}")
                        print(f"    {cand_text}")
                    return {
                        'success'        : True,
                        'best_text'      : cand_text,
                        'best_score'     : S,
                        'round_found'    : k,
                        'score_breakdown': result,
                        'beam_history'   : beam_history,
                    }

                round_candidates.append(
                    Beam(score=S, text=cand_text, round_no=k,
                         history=beam.history)
                )

        if not round_candidates:
            break

        # Prune to top-B beams
        round_candidates.sort(reverse=True)
        beams = round_candidates[:beam_width]

        top = beams[0]
        beam_history.append({'round': k, 'top_score': top.score, 'top_text': top.text})

        if verbose:
            print(f"  top S={top.score:.4f}  |  {top.text[:80]}...")

        if top.score > best_score:
            best_score  = top.score
            best_attack = top.text

    # Loop exhausted without misclassification
    final_result = composite_score(original_text, best_attack or original_text, **weights)
    return {
        'success'        : False,
        'best_text'      : best_attack or original_text,
        'best_score'     : best_score,
        'round_found'    : None,
        'score_breakdown': final_result,
        'beam_history'   : beam_history,
    }
def run_csbp_batch(
    texts          : List[str],
    labels         : List[str],
    classifier_fn  : Callable[[str], str],
    K              : int = 5,
    beam_width     : int = 5,
    n_candidates   : int = 10,
    verbose        : bool = False,
) -> List[dict]:
    """
    Run CSBP over a list of (text, label) pairs.
    Returns list of result dicts, one per input.
    """
    results = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        print(f"\n[{i+1}/{len(texts)}] label={label}  text={text[:60]}...")
        result = csbp_loop(
            original_text   = text,
            original_label  = label,
            classifier_fn   = classifier_fn,
            K               = K,
            beam_width      = beam_width,
            n_candidates    = n_candidates,
            verbose         = verbose,
        )
        result['original_text']  = text
        result['original_label'] = label
        results.append(result)

    n_success = sum(r['success'] for r in results)
    print(f"\n=== ASR: {n_success}/{len(results)} ({100*n_success/len(results):.1f}%) ===")
    return results
if __name__ == "__main__":

    # Plug in any real classifier here — dummy shown for testing
    def dummy_classifier(text: str) -> str:
        """Always returns 'positive' unless the word 'bad' appears."""
        return 'negative' if 'bad' in text.lower() else 'positive'

    original  = "The film was genuinely bad and quite disappointing overall."
    label     = dummy_classifier(original)

    print(f"Original label : {label}")
    print(f"Original text  : {original}\n")

    result = csbp_loop(
        original_text   = original,
        original_label  = label,
        classifier_fn   = dummy_classifier,
        K               = 5,
        beam_width      = 3,
        n_candidates    = 8,
        verbose         = True,
    )

    print("\n--- Final Result ---")
    print(f"Success      : {result['success']}")
    print(f"Round found  : {result['round_found']}")
    print(f"Best score S : {result['best_score']:.4f}")
    print(f"Best text    : {result['best_text']}")
    print(f"Score terms  : {result['score_breakdown']}")
