from typing import List
from enum import Enum
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


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
    def __init__(self, data: List[Features], labels: List[float], test_ratio: float):
        if not (0.0 < test_ratio < 1.0):
            raise ValueError("test_ratio must be between 0 and 1")
        
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            data, labels, test_size=test_ratio, random_state=67
        )

    def train_size(self) -> int:
        return len(self.train_data)
    
    def test_size(self) -> int:
        return len(self.test_data)