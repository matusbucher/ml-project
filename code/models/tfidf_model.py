from typing import Callable, Dict, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from models.model_interface import ModelInterface
from normalized_data import *
from utils import *


class TfIdfModel(ModelInterface):
    @staticmethod
    def __default_preprocess(text: str) -> str:
        return normalize_whitespaces(insert_spaces(text.lower()))

    def __init__(self, normalized_data: NormalizedData = None,
                 bounding_func: Callable[[float], float] = identity,
                 vectorizer: TfidfVectorizer = TfidfVectorizer(
                     ngram_range=(1, 2),
                     sublinear_tf=True,
                     stop_words="english",
                     norm="l2",
                     min_df=3,
                     max_df=0.3
                 ),
                 regression_model = SVR(
                     kernel="linear",
                     C=0.1,
                     epsilon=0.1
                 ),
                 preprocess: Optional[Callable[[str], str]] = None) -> None:
        
        self._model = Pipeline([
            ("tfidf", vectorizer),
            ("regressor", regression_model)
        ])
        
        self._bounding_func = bounding_func
        self._preprocess = preprocess if preprocess is not None else self.__default_preprocess

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)
    
    def fit(self, data: List[Features], labels: List[float]) -> None:
        X = [self._preprocess(d.description) for d in data]
        self._model.fit(X, labels)

    def search_fit(self, data: List[Features], labels: List[float], vectorizer_params: Optional[Dict[str, List[float]]] = None, regressor_params: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
        X = [self._preprocess(d.description) for d in data]

        new_params = {}
        if vectorizer_params is not None:
            new_params.update({f"tfidf__{key}": value for key, value in vectorizer_params.items()})
        if regressor_params is not None:
            new_params.update({f"regressor__{key}": value for key, value in regressor_params.items()})

        search = GridSearchCV(self._model, new_params, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
        search.fit(X, labels)

        self._model = search.best_estimator_
        return search.best_params_
    
    def predict(self, features: Features) -> float:
        X = [self._preprocess(features.description)]
        return self._bounding_func(self._model.predict(X)[0])