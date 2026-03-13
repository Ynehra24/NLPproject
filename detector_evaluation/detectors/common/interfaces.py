"""Detector interfaces and serializable result structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DetectorResult:
    id: str
    detector_name: str
    ai_score: float
    predicted_label: str
    threshold_used: float
    metadata: Optional[Dict] = None


class BaseDetector(ABC):
    detector_name: str

    @abstractmethod
    def score_texts(self, texts: List[str]) -> List[float]:
        """Return AI-likelihood scores in [0, 1], one per text."""

    def predict(self, texts: List[str], threshold: float = 0.5) -> List[str]:
        scores = self.score_texts(texts)
        return ["ai" if s >= threshold else "human" for s in scores]
