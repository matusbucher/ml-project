from typing import List, Dict
import numpy as np
from sklearn.linear_model import LinearRegression
from nltk.stem.porter import PorterStemmer

from models.model_interface import ModelInterface
from normalized_data import *
from utils import *

class BagOfWordsModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None):
        self._stopwords = load_stopwords()
        self._stemmer = PorterStemmer()
        self._vocab: Dict[str, int] = {}
        self._reg = LinearRegression()

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)
    
    def fit(self, data: List[Features], labels: List[float]) -> None:
        processed_data = self.__preprocess_data(data)
        self._vocab = self.__build_vocab(processed_data)
        X = self.__vectorize_data(processed_data)
        y = np.array(labels)
        self._reg.fit(X, y)
    
    def predict(self, features: Features) -> float:
        processed_features = self.__preprocess_data([features])[0]
        X = self.__vectorize_data([processed_features])
        return bound_target(self._reg.predict(X)[0])
    
    def vocab_size(self) -> int:
        return len(self._vocab)

    @classmethod
    def __build_vocab(cls, data: List[List[str]]) -> Dict[str, int]:
        words = set(w for d in data for w in d)
        return {w: i for i, w in enumerate(words)}

    def __preprocess_data(self, data: List[Features]) -> List[List[str]]:
        result : List[List[str]] = []
        for features in data:
            words = remove_latex(features.description).split()
            lowercase_words(words)
            filter_words(words, self._stopwords, contains=False)
            stem_words(words, self._stemmer.stem)
            result.append(words)
        return result

    def __vectorize_data(self, data: List[List[str]]) -> np.ndarray:
        vectors = np.zeros((len(data), len(self._vocab)))
        for i, words in enumerate(data):
            for w in words:
                j = self._vocab.get(w, -1)
                if j != -1:
                    vectors[i][j] += 1
        return vectors