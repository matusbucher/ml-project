import math_lighteval
import baseline_models

if __name__ == "__main__":
    data = math_lighteval.data_load(normalize_labels=True)
    baseline = baseline_models.BaselineAverageModel(data)
    mse = baseline.evaluate()
    print(f"Baseline Average Model MSE: {mse}")