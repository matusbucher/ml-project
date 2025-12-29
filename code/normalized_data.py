from dataclasses import dataclass
from enum import Enum
from typing import List


class ProblemType(Enum):
    ALGEBRA = "algebra"
    INTERMEDIATE_ALGEBRA = "intermediate algebra"
    PREALGEBRA = "prealgebra"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number theory"
    COUNTING_AND_PROBABILITY = "counting and probability"
    PRECALCULUS = "precalculus"


@dataclass
class Features():
    description: str
    solution: str
    problem_type: ProblemType


class NormalizedData:
    def __init__(self, data: List[Features], labels: List[float], split_ratio: float):
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio must be between 0 and 1")
        
        split_index = int(len(data) * split_ratio)
        self.train_data = data[:split_index]
        self.train_labels = labels[:split_index]
        self.test_data = data[split_index:]
        self.test_labels = labels[split_index:]

    def train_size(self) -> int:
        return len(self.train_data)
    
    def test_size(self) -> int:
        return len(self.test_data)