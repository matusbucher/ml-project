from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from models.model_interface import ModelInterface
from normalized_data import *
from utils import *


class TfIdfModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None, bounding_func : Callable[[float], float] = identity, regression_model = Ridge()):
        if regression_model is None:
            regression_model = Ridge()
        
        self._model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                sublinear_tf=True,
                lowercase=True,
                norm="l2"
            )),
            ("regressor", regression_model)
        ])
        
        self._bounding_func = bounding_func

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)
    
    def fit(self, data: List[Features], labels: List[float]) -> None:
        X = self.__preprocess_data(data)
        self._model.fit(X, labels)

    def search_fit(self, data: List[Features], labels: List[float], params : Dict[str, List[float]]) -> Dict[str, float]:
        X = self.__preprocess_data(data)
        new_params = {f"regressor__{key}": value for key, value in params.items()}

        search = GridSearchCV(self._model, new_params, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
        search.fit(X, labels)

        self._model = search.best_estimator_
        return search.best_params_
    
    def predict(self, features: Features) -> float:
        X = self.__preprocess_data([features])
        return self._bounding_func(self._model.predict(X)[0])

    def __preprocess_data(self, data: List[Features]) -> List[str]:
        return [remove_tex(d.description).lower() for d in data]