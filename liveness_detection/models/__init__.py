
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class GridResult:

    variance: float
    position: Tuple[int, int]
    size: Tuple[int, int]
    is_face_region: bool = True
    region: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ProcessingResult:
    is_live: bool
    confidence: float
    max_variance: float
    min_variance: float
    avg_variance: float
    anomaly_score: float
    grid_count: int
    execution_time: float
    details: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    method_name: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    total_images: int
    total_grids: int
    accuracy: float = 0.0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def precision(self) -> float:
        """Calculate precision."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Calculate recall."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
