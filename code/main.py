import math_lighteval
import easy2hard_bench
import baseline_models

if __name__ == "__main__":
    # data = math_lighteval.data_load()
    data = easy2hard_bench.data_load()

    model1 = baseline_models.BaselineRandomModel(data)
    model2 = baseline_models.BaselineAverageModel(data)
    model3 = baseline_models.BaselineDescriptionLengthModel(data)
    model4 = baseline_models.BaselineSolutionLengthModel(data)

    print("Baseline Random Model - Test MSE:", model1.test_accuracy())
    print("Baseline Average Model - Test MSE:", model2.test_accuracy())
    print("Baseline Description Length Model - Test MSE:", model3.test_accuracy())
    print("Baseline Solution Length Model - Test MSE:", model4.test_accuracy())
