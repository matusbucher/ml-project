from abc import ABC, abstractmethod
from typing import List
from sklearn.metrics import root_mean_squared_error

from normalized_data import Features

class ModelInterface(ABC):
    @abstractmethod
    def fit(self, data: List[Features], labels: List[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: Features) -> float:
        raise NotImplementedError

    def score(self, data: List[Features], labels: List[float]) -> float:
        predictions = [self.predict(f) for f in data]
        return root_mean_squared_error(labels, predictions)