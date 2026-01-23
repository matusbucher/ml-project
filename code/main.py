from typing import List, Tuple

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

import easy2hard_bench
import math_lighteval
from normalized_data import NormalizedData
from models.model_interface import ModelInterface
from models.baseline import RandomModel, AverageModel, DescriptionLengthModel, SolutionLengthModel
from models.tfidf_model import TfIdfModel
from models.ling_feature_model import LingFeatureModel
from utils import *


TFIDF_SEARCH_PARAMS = {
    "min_df": [1, 3, 5, 10, 15],
    "max_df": [0.1, 0.3, 0.5, 0.7, 0.9],
}

RIDGE_SEARCH_PARAMS = {
    "alpha": [0.1, 1.0, 10.0]
}

SVR_SEARCH_PARAMS = {
    "C": [0.01, 0.1, 1.0],
    "epsilon": [0.01, 0.1, 1.0]
}

RF_SEARCH_PARAMS = {
    "n_estimators": [300, 500, 700],
    "max_depth": [5, 10, 20],
    "min_samples_split": [10, 20, 50],
    "min_samples_leaf": [3, 5, 10],
    "max_features": [0.6, 0.8, 1.0],
}

GBR_SEARCH_PARAMS = {
    "n_estimators": [300, 500, 700],
    "learning_rate": [0.01, 0.1, 0.5],
    "max_depth": [5, 10, 20],
    "min_samples_split": [10, 20, 50],
    "min_samples_leaf": [3, 5, 10],
    "subsample": [0.6, 0.8, 1.0],
}


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


def tfidf_fine_tune(data: NormalizedData, preprocess: Callable[[str], str]) -> None:
    ridge_model = TfIdfModel(bounding_func=clip, regression_model=Ridge(alpha=1.0), preprocess=preprocess)
    svr_model = TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear", C=0.1, epsilon=0.1), preprocess=preprocess)
    best_ridge = ridge_model.search_fit(data.train_data, data.train_labels, vectorizer_params=TFIDF_SEARCH_PARAMS, regressor_params=None)
    best_svr = svr_model.search_fit(data.train_data, data.train_labels, vectorizer_params=TFIDF_SEARCH_PARAMS, regressor_params=None)

    print("Best Ridge parameters:", best_ridge)
    print_scores("TF-IDF (Ridge) Model", ridge_model, data)
    print("Best SVR parameters:", best_svr)
    print_scores("TF-IDF (Linear SVR) Model", svr_model, data)


def ling_feature_fine_tune(data: NormalizedData) -> None:
    rf_model = LingFeatureModel(regressor=RandomForestRegressor(bootstrap=True, random_state=67))
    rf_best_params = rf_model.search_fit(data.train_data, data.train_labels, params=RF_SEARCH_PARAMS)
    gb_model = LingFeatureModel(regressor=GradientBoostingRegressor(loss="squared_error", max_features=None, random_state=67))
    gb_best_params = gb_model.search_fit(data.train_data, data.train_labels, params=GBR_SEARCH_PARAMS)

    print("Best Random Forest parameters:", rf_best_params)
    print_scores("LingFeature Random Forest Model", rf_model, data)
    print("Best Gradient Boosting parameters:", gb_best_params)
    print_scores("LingFeature Gradient Boosting Model", gb_model, data)


def ling_feature_importances(data: NormalizedData) -> None:
    models = ling_feature_models()
    for model_name, model in models:
        model.fit(data.train_data, data.train_labels)
        importances = model.feature_importances()
        if importances is not None:
            print(f"\nFeature importances for {model_name}:")
            for feature_name, importance in importances.items():
                print(f"  {feature_name}: {importance:.4f}")
        else:
            print(f"Model {model_name} does not provide feature importances.")


def baseline_models() -> List[Tuple[str, ModelInterface]]:
    return [
        ("Random", RandomModel(bounding_func=clip)),
        ("Average", AverageModel(bounding_func=clip)),
        ("Description Length", DescriptionLengthModel(bounding_func=clip)),
        ("Solution Length", SolutionLengthModel(bounding_func=clip))
    ]


def tfidf_models() -> List[Tuple[str, ModelInterface]]:
    return [
        ("Ridge (A)", TfIdfModel(bounding_func=clip, regression_model=Ridge(alpha=1.0), preprocess=preprocess_a)),
        ("Ridge (B)", TfIdfModel(bounding_func=clip, regression_model=Ridge(alpha=1.0), preprocess=preprocess_b)),
        ("Ridge (C)", TfIdfModel(bounding_func=clip, regression_model=Ridge(alpha=1.0), preprocess=prerocess_c)),
        ("SVR (A)", TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear", C=0.1, epsilon=0.1), preprocess=preprocess_a)),
        ("SVR (B)", TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear", C=0.1, epsilon=0.1), preprocess=preprocess_b)),
        ("SVR (C)", TfIdfModel(bounding_func=clip, regression_model=SVR(kernel="linear", C=0.1, epsilon=0.1), preprocess=prerocess_c)),
    ]


def ling_feature_models() -> List[Tuple[str, ModelInterface]]:
    return [
        ("LingFeature Random Forest (MATH-best)", LingFeatureModel(regressor=RandomForestRegressor(bootstrap=True, random_state=67, n_estimators=300, max_depth=20, min_samples_split=10, min_samples_leaf=3, max_features=0.6))),
        ("LingFeature Random Forest (E2H-best)", LingFeatureModel(regressor=RandomForestRegressor(bootstrap=True, random_state=67, n_estimators=500, max_depth=10, min_samples_split=10, min_samples_leaf=3, max_features=0.8))),
        ("LingFeature Gradient Boosting (MATH-best)", LingFeatureModel(regressor=GradientBoostingRegressor(loss="squared_error", max_features=None, random_state=67, n_estimators=300, learning_rate=0.01, max_depth=10, min_samples_split=20, min_samples_leaf=3, subsample=0.6))),
        ("LingFeature Gradient Boosting (E2H-best)", LingFeatureModel(regressor=GradientBoostingRegressor(loss="squared_error", max_features=None, random_state=67, n_estimators=700, learning_rate=0.01, max_depth=5, min_samples_split=50, min_samples_leaf=10, subsample=0.8)))
    ]


if __name__ == "__main__":
    data1 = math_lighteval.data_load()
    data2 = easy2hard_bench.data_load()

    models = []

    print("=== Math LightEval Dataset ===")
    test_models_on_dataset(models, data1)

    print("\n=== Easy2Hard Benchmark Dataset ===")
    test_models_on_dataset(models, data2)