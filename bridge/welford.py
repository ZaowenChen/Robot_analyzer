"""
Welford's Online Variance Algorithm — single-pass mean/variance with O(1) memory.
"""

import math
from dataclasses import dataclass


@dataclass
class WelfordAccumulator:
    """Single-pass mean/variance computation with O(1) memory."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')

    def update(self, value: float):
        if math.isnan(value) or math.isinf(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.min_val, 6) if self.min_val != float('inf') else None,
            "max": round(self.max_val, 6) if self.max_val != float('-inf') else None,
            "count": self.count,
        }
