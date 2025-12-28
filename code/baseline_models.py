from normalized_data import *


class BaselineAverageModel:
    def __init__(self, normalized_data: NormalizedData):
        self.normalized_data = normalized_data
        self.average_label = sum(normalized_data.train_labels) / normalized_data.train_size()

    def predict(self, features: Features) -> float:
        return self.average_label

    def evaluate(self) -> float:
        predictions = [self.predict(f) for f in self.normalized_data.test_data]
        mse = sum((pred - true) ** 2 for pred, true in zip(predictions, self.normalized_data.test_labels)) / self.normalized_data.test_size()
        return mse