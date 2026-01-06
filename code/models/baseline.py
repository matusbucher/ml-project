from typing import List
import numpy as np
from sklearn.linear_model import LinearRegression

from models.model_interface import ModelInterface
from normalized_data import *
from utils import *


class BaselineRandomModel(ModelInterface):
    def fit(self, data: List[Features], labels: List[float]) -> None:
        pass

    def predict(self, features: Features) -> float:
        return np.random.uniform(0, 1)


class BaselineAverageModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None):
        self._average_label : float = 0.0

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)

    def fit(self, data: List[Features], labels: List[float]) -> None:
        self._average_label = sum(labels) / len(labels)
        
    def predict(self, features: Features) -> float:
        return self._average_label


class BaselineDescriptionLengthModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None):
        self._reg = LinearRegression()

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)
            return
    
    def fit(self, data: List[Features], labels: List[float]) -> None:
        X = np.array([[len(f.description)] for f in data])
        y = np.array(labels)
        self._reg.fit(X, y)
    
    def predict(self, features: Features) -> float:
        X = np.array([[len(features.description)]])
        return bound_target(self._reg.predict(X)[0])


class BaselineSolutionLengthModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None):
        self._reg = LinearRegression()

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)
            return
    
    def fit(self, data: List[Features], labels: List[float]) -> None:
        X = np.array([[len(f.solution)] for f in data])
        y = np.array(labels)
        self._reg.fit(X, y)

    def predict(self, features: Features) -> float:
        X = np.array([[len(features.solution)]])
        return bound_target(self._reg.predict(X)[0])