import math_lighteval
import easy2hard_bench
import models.baseline
import models.bag_of_words

if __name__ == "__main__":
    data = math_lighteval.data_load()
    # data = easy2hard_bench.data_load()

    # model1 = models.baseline.BaselineRandomModel()
    # model2 = models.baseline.BaselineAverageModel(data)
    # model3 = models.baseline.BaselineDescriptionLengthModel(data)
    # model4 = models.baseline.BaselineSolutionLengthModel(data)

    # print("Baseline Random Model - Test score:", model1.rsme(data.test_data, data.test_labels))
    # print("Baseline Average Model - Test score:", model2.rsme(data.test_data, data.test_labels))
    # print("Baseline Description Length Model - Test score:", model3.rsme(data.test_data, data.test_labels))
    # print("Baseline Solution Length Model - Test score:", model4.rsme(data.test_data, data.test_labels))

    bag_model = models.bag_of_words.BagOfWordsModel(data)
    print("Bag of Words Model - Train score:", bag_model.rmse(data.train_data, data.train_labels))
    print("Bag of Words Model - Test score:", bag_model.rmse(data.test_data, data.test_labels))