from abc import ABC, abstractmethod
from typing import List
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from normalized_data import Features

class ModelInterface(ABC):
    @abstractmethod
    def fit(self, data: List[Features], labels: List[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: Features) -> float:
        raise NotImplementedError

    def r2_score(self, data: List[Features], labels: List[float]) -> float:
        predictions = [self.predict(f) for f in data]
        return r2_score(labels, predictions)
    
    def rmse(self, data: List[Features], labels: List[float]) -> float:
        predictions = [self.predict(f) for f in data]
        return root_mean_squared_error(labels, predictions)
    
    def mae(self, data: List[Features], labels: List[float]) -> float:
        predictions = [self.predict(f) for f in data]
        return mean_absolute_error(labels, predictions)
    
    def get_metrics(self, data: List[Features], labels: List[float]) -> dict:
        predictions = [self.predict(f) for f in data]
        return {
            "r2": r2_score(labels, predictions),
            "rmse": root_mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions)
        }