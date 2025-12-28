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
    def __init__(self, train_data: List[Features], train_labels: List[float], test_data: List[Features], test_labels: List[float]):
        if len(train_data) != len(train_labels):
            raise ValueError("Number of training samples and training labels must be the same.")
        
        if len(test_data) != len(test_labels):
            raise ValueError("Number of test samples and test labels must be the same.")

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def train_size(self) -> int:
        return len(self.train_data)
    
    def test_size(self) -> int:
        return len(self.test_data)