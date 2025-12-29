import math_lighteval
import baseline_models

if __name__ == "__main__":
    data = math_lighteval.data_load(normalize_labels=True)
    baseline = baseline_models.BaselineDescriptionLengthModel(data)
    print("Train MSE:", baseline.train_accuracy())
    print("Test MSE:", baseline.test_accuracy())