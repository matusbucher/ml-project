import numpy as np
from sklearn.linear_model import LinearRegression

from normalized_data import *


class BaselineAverageModel:
    def __init__(self, normalized_data: NormalizedData):
        self.normalized_data = normalized_data
        self.average_label = sum(normalized_data.train_labels) / normalized_data.train_size()

    def predict(self, features: Features) -> float:
        return self.average_label
    
    def train_accuracy(self) -> float:
        predictions = [self.predict(f) for f in self.normalized_data.train_data]
        mse = sum((pred - true) ** 2 for pred, true in zip(predictions, self.normalized_data.train_labels)) / self.normalized_data.train_size()
        return mse

    def test_accuracy(self) -> float:
        predictions = [self.predict(f) for f in self.normalized_data.test_data]
        mse = sum((pred - true) ** 2 for pred, true in zip(predictions, self.normalized_data.test_labels)) / self.normalized_data.test_size()
        return mse


class BaselineDescriptionLengthModel:
    def __init__(self, normalized_data: NormalizedData):
        self.normalized_data = normalized_data
        self.X_train = np.array([len(f.description) for f in normalized_data.train_data]).reshape(-1, 1)
        self.X_test = np.array([len(f.description) for f in normalized_data.test_data]).reshape(-1, 1)
        self.y_train = np.array(normalized_data.train_labels)
        self.y_test = np.array(normalized_data.test_labels)
        
        self.reg = LinearRegression().fit(self.X_train, self.y_train)

    def predict(self, features: Features) -> float:
        return self.reg.predict(np.array([[len(features.description)]]))[0]

    def train_accuracy(self) -> float:
        predictions = self.reg.predict(self.X_train)
        mse = np.mean((predictions - self.y_train) ** 2)
        return mse

    def test_accuracy(self) -> float:
        predictions = self.reg.predict(self.X_test)
        mse = np.mean((predictions - self.y_test) ** 2)
        return mse
