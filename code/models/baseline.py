from typing import List
import numpy as np
from sklearn.linear_model import LinearRegression

from models.model_interface import ModelInterface
from normalized_data import *
from utils import *


class RandomModel(ModelInterface):
    def fit(self, data: List[Features], labels: List[float]) -> None:
        pass

    def predict(self, features: Features) -> float:
        return np.random.uniform(0, 1)


class AverageModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None):
        self._average_label: float = 0.0

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)

    def fit(self, data: List[Features], labels: List[float]) -> None:
        self._average_label = sum(labels) / len(labels)
        
    def predict(self, features: Features) -> float:
        return self._average_label


class DescriptionLengthModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None):
        self._reg = LinearRegression()

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)
            return
    
    def fit(self, data: List[Features], labels: List[float]) -> None:
        X = [[len(f.description)] for f in data]
        self._reg.fit(X, labels)
    
    def predict(self, features: Features) -> float:
        X = [[len(features.description)]]
        return clip(self._reg.predict(X)[0])


class SolutionLengthModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None):
        self._reg = LinearRegression()

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)
            return
    
    def fit(self, data: List[Features], labels: List[float]) -> None:
        X = [[len(f.solution)] for f in data]
        self._reg.fit(X, labels)

    def predict(self, features: Features) -> float:
        X = [[len(features.solution)]]
        return clip(self._reg.predict(X)[0])