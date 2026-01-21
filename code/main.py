from typing import List, Tuple

from sklearn.linear_model import Ridge
from sklearn.svm import SVR

import easy2hard_bench
import math_lighteval
from utils import clip, sigmoid
from normalized_data import NormalizedData
from models.model_interface import MetricType, ModelInterface
from models.baseline import DescriptionLengthModel
from models.tfidf_model import TfIdfModel
from models.ling_feature_model import LingFeatureModel


RIDGE_PARAMS = {
    "alpha": [0.1, 1.0, 10.0]
}

SVR_PARAMS = {
    "C": [0.1, 1.0, 10.0],
    "epsilon": [0.01, 0.1, 1.0]
}

MODELS = [
    ("Description Length Baseline", DescriptionLengthModel()),
    ("TF-IDF (Ridge) Model", TfIdfModel(bounding_func=clip, regression_model=Ridge(alpha=1.0))),
    ("TF-IDF (Linear SVR) Model", TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear", C=0.1, epsilon=0.1))),
    ("Linguistic Feature Model", LingFeatureModel(bounding_func=clip)),
]

# GradientBoostingRegressor(
#             loss="squared_error",
#             learning_rate=0.03,
#             n_estimators=500,
#             max_depth=5,
#             min_samples_split=20,
#             min_samples_leaf=5,
#             subsample=0.9,
#             max_features=None,
#             random_state=67
#         )

def print_scores(model_name: str, model: ModelInterface, data: NormalizedData, metric: MetricType) -> None:
    print(f"{model_name} model scores:")
    print(f"  train: {model.get_metrics(data.train_data, data.train_labels)[metric]}")
    print(f"  test: {model.get_metrics(data.test_data, data.test_labels)[metric]}")

def test_models_on_dataset(models: List[Tuple[str, ModelInterface]], data: NormalizedData, metric : MetricType) -> None:
    print("TRAIN SIZE:", len(data.train_data))
    print("TEST SIZE:", len(data.test_data))

    for model_name, model in models:
        model.fit(data.train_data, data.train_labels)
        print_scores(model_name, model, data, metric)

def tfidf_search_params(data: NormalizedData) -> None:
    ridge_params = {
        "alpha": [0.1, 1.0, 10.0]
    }
    svr_params = {
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.01, 0.1, 1.0]
    }

    ridge_model = TfIdfModel(bounding_func=clip, regression_model=Ridge())
    best_ridge1 = ridge_model.search_fit(data.train_data, data.train_labels, ridge_params)
    svr_model = TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear"))
    best_svr1 = svr_model.search_fit(data.train_data, data.train_labels, svr_params)

    print_scores("TF-IDF (Ridge) Model", ridge_model, data, MetricType.RMSE)
    print_scores("TF-IDF (Linear SVR) Model", svr_model, data, MetricType.RMSE)
    print("Best Ridge parameters:", best_ridge1)
    print("Best SVR parameters:", best_svr1)


if __name__ == "__main__":
    data1 = math_lighteval.data_load()
    data2 = easy2hard_bench.data_load()

    print("=== Math LightEval Dataset ===")
    # tfidf_search_params(data1)
    test_models_on_dataset(MODELS, data1, MetricType.RMSE)

    print("\n=== Easy2Hard Benchmark Dataset ===")
    # tfidf_search_params(data2)
    test_models_on_dataset(MODELS, data2, MetricType.RMSE)