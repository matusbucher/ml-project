from abc import ABC, abstractmethod
from typing import List, Dict
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sqlalchemy import Enum

from normalized_data import Features


class MetricType(Enum):
    R2 = "r2"
    RMSE = "rmse"
    

class ModelInterface(ABC):
    @abstractmethod
    def fit(self, data: List[Features], labels: List[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, features: Features) -> float:
        raise NotImplementedError

    def r2(self, data: List[Features], labels: List[float]) -> float:
        predictions = [self.predict(f) for f in data]
        return r2_score(labels, predictions)
    
    def rmse(self, data: List[Features], labels: List[float]) -> float:
        predictions = [self.predict(f) for f in data]
        return root_mean_squared_error(labels, predictions)
    
    def mae(self, data: List[Features], labels: List[float]) -> float:
        predictions = [self.predict(f) for f in data]
        return mean_absolute_error(labels, predictions)
    
    def get_metrics(self, data: List[Features], labels: List[float]) -> Dict[MetricType, float]:
        predictions = [self.predict(f) for f in data]
        return {
            MetricType.R2: round(r2_score(labels, predictions), 3),
            MetricType.RMSE: round(root_mean_squared_error(labels, predictions), 3),
        }