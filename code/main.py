from typing import List, Tuple

from sklearn.linear_model import Ridge
from sklearn.svm import SVR

import easy2hard_bench
import math_lighteval
from normalized_data import NormalizedData
from models.model_interface import MetricType, ModelInterface
from models.baseline import DescriptionLengthModel
from models.tfidf_model import TfIdfModel
from models.ling_feature_model import LingFeatureModel
from utils import *


TFIDF_PARAMS = {
    "min_df": [1, 3, 5, 10, 15],
    "max_df": [0.1, 0.3, 0.5, 0.7, 0.9],
}

RIDGE_PARAMS = {
    "alpha": [0.1, 1.0, 10.0]
}

SVR_PARAMS = {
    "C": [0.01, 0.1, 1.0],
    "epsilon": [0.01, 0.1, 1.0]
}

# GradientBoostingRegressor(
#     loss="squared_error",
#     learning_rate=0.03,
#     n_estimators=500,
#     max_depth=5,
#     min_samples_split=20,
#     min_samples_leaf=5,
#     subsample=0.9,
#     max_features=None,
#     random_state=67
# )


def preprocess_a(text: str) -> str:
    return normalize_whitespaces(insert_spaces(text.lower()))

def preprocess_b(text: str) -> str:
    return normalize_whitespaces(insert_spaces(remove_tex(text.lower())))

def prerocess_c(text: str) -> str:
    return normalize_whitespaces(insert_spaces(remove_tex_symbols(text.lower())))

def print_scores(model_name: str, model: ModelInterface, data: NormalizedData) -> None:
    print(f"{model_name} model scores:")
    print(f"  train: {model.get_metrics(data.train_data, data.train_labels)}")
    print(f"  test: {model.get_metrics(data.test_data, data.test_labels)}")

def test_models_on_dataset(models: List[Tuple[str, ModelInterface]], data: NormalizedData) -> None:
    print("TRAIN SIZE:", len(data.train_data))
    print("TEST SIZE:", len(data.test_data))

    for model_name, model in models:
        model.fit(data.train_data, data.train_labels)
        print_scores(model_name, model, data)


def tfidf_search_params(data: NormalizedData, preprocess: Callable[[str], str]) -> None:
    ridge_model = TfIdfModel(bounding_func=clip, regression_model=Ridge(alpha=1.0), preprocess=preprocess)
    best_ridge = ridge_model.search_fit(data.train_data, data.train_labels, vectorizer_params=TFIDF_PARAMS, regressor_params=None)
    svr_model = TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear", C=0.1, epsilon=0.1), preprocess=preprocess)
    best_svr = svr_model.search_fit(data.train_data, data.train_labels, vectorizer_params=TFIDF_PARAMS, regressor_params=None)

    print_scores("TF-IDF (Ridge) Model", ridge_model, data)
    print_scores("TF-IDF (Linear SVR) Model", svr_model, data)
    print("Best Ridge parameters:", best_ridge)
    print("Best SVR parameters:", best_svr)


if __name__ == "__main__":
    data1 = math_lighteval.data_load()
    data2 = easy2hard_bench.data_load()

    models = [
        ("Ridge (A)", TfIdfModel(bounding_func=clip, regression_model=Ridge(alpha=1.0), preprocess=preprocess_a)),
        ("Ridge (B)", TfIdfModel(bounding_func=clip, regression_model=Ridge(alpha=1.0), preprocess=preprocess_b)),
        ("Ridge (C)", TfIdfModel(bounding_func=clip, regression_model=Ridge(alpha=1.0), preprocess=prerocess_c)),
        ("SVR (A)", TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear", C=0.1, epsilon=0.1), preprocess=preprocess_a)),
        ("SVR (B)", TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear", C=0.1, epsilon=0.1), preprocess=preprocess_b)),
        ("SVR (C)", TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear", C=0.1, epsilon=0.1), preprocess=prerocess_c)),
    ]

    print("=== Math LightEval Dataset ===")
    test_models_on_dataset(models, data1)

    print("\n=== Easy2Hard Benchmark Dataset ===")
    test_models_on_dataset(models, data2)