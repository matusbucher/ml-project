from typing import List, Dict
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from nltk.stem.porter import PorterStemmer

from models.model_interface import ModelInterface
from normalized_data import *
from utils import *

class BagOfWordsModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None):
        self._stopwords = load_stopwords()
        self._stemmer = PorterStemmer()
        self._vocab: Dict[str, int] = {}
        self._model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                sublinear_tf=True,
                lowercase=True,
                norm="l2"
            )),
            ("ridge", Ridge())
        ])

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)
    
    def fit(self, data: List[Features], labels: List[float]) -> None:
        X = self.__preprocess_data(data)
        y = np.array(labels)
        self._model.fit(X, y)

    def search_fit(self, data: List[Features], labels: List[float], alphas: List[float]) -> float:
        X = self.__preprocess_data(data)
        y = np.array(labels)

        params = {"ridge__alpha": alphas}
        search = GridSearchCV(self._model, params, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
        search.fit(X, y)

        self._model = search.best_estimator_
        return search.best_params_["ridge__alpha"]
    
    def predict(self, features: Features) -> float:
        X = self.__preprocess_data([features])
        return bound_target(self._model.predict(X)[0])

    def __preprocess_data(self, data: List[Features]) -> List[str]:
        return [remove_latex(d.description).lower() for d in data]
