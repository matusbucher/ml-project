from __future__ import annotations
from typing import List
from enum import Enum
from dataclasses import dataclass
import random
from sklearn.model_selection import train_test_split


@dataclass
class Features():
    description: str
    solution: str


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
    
    @staticmethod
    def merge_data(data_list: List[NormalizedData], test_ratio: float, shuffle: bool = True, random_state: int = 67) -> NormalizedData:
        merged_data = []
        merged_labels = []

        for nd in data_list:
            merged_data.extend(nd.train_data)
            merged_data.extend(nd.test_data)
            merged_labels.extend(nd.train_labels)
            merged_labels.extend(nd.test_labels)
        
        if shuffle:
            combined = list(zip(merged_data, merged_labels))
            random.Random(random_state).shuffle(combined)
            merged_data[:], merged_labels[:] = zip(*combined)

        return NormalizedData(merged_data, merged_labels, test_ratio)