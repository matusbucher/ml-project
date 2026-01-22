from typing import Dict, List, Optional, Callable
import re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from models.model_interface import ModelInterface
from normalized_data import *
from utils import *


TEX_OPERATORS = ["\\frac", "\\dfrac", "\\sum", "\\lim", "\\sin", "\\cos", "\\tan", "\\log", "\\ln", "\\sqrt", "\\leq", "\\le", "\\geq", "\\ge", "\\neq", "+", "-", "*", "/", "=", "<", ">", "^"]


class LingFeatures:
    def __init__(self, input_text: str):
        text = input_text.lower()
        text_no_tex = remove_tex(text)
        words_no_tex = text_no_tex.split()
        only_alpha_words = LingFeatures.__only_alpha_words(words_no_tex)
        math = LingFeatures.__extract_math_expressions(input_text)

        self.num_words = len(only_alpha_words)
        self.avg_word_length = sum(len(word) for word in only_alpha_words) / self.num_words if self.num_words > 0 else 0
        self.num_sentences = LingFeatures.__number_of_sentences(text_no_tex)
        self.avg_sentence_length = self.num_words / self.num_sentences if self.num_sentences > 0 else 0
        self.num_syllabes = LingFeatures.__number_of_syllables(only_alpha_words)
        self.avg_syllables_per_word = self.num_syllabes / self.num_words if self.num_words > 0 else 0

        self.flesch_reading_ease = LingFeatures.__flesch_reading_ease(self.avg_sentence_length, self.avg_syllables_per_word)
        self.flesch_kincaid_grade_level = LingFeatures.__flesch_kincaid_grade_level(self.avg_sentence_length, self.avg_syllables_per_word)

        self.num_numbers = LingFeatures.__number_of_numbers(text.split())
        self.num_operators = LingFeatures.__number_of_operators(math)
        self.num_variables = LingFeatures.__number_of_variables(math)

    def extract_features(self) -> List[float]:
        return [
            self.num_words,
            self.avg_word_length,
            self.num_sentences,
            self.avg_sentence_length,
            self.num_syllabes,
            self.avg_syllables_per_word,
            self.flesch_reading_ease,
            self.flesch_kincaid_grade_level,
            self.num_numbers,
            self.num_operators,
            self.num_variables
        ]
    
    @staticmethod
    def __only_alpha_words(words: List[str]) -> List[str]:
        result = []
        for word in words:
            if word.isalpha():
                result.append(word)
                continue

            cleaned_word = re.sub(r"[^a-zA-Z-]", "", word)
            if not cleaned_word:
                continue

            if cleaned_word.find("-") != -1:
                sub_words = cleaned_word.split("-")
                for sub_word in sub_words:
                    if sub_word:
                        result.append(sub_word)
                continue

            result.append(cleaned_word)

        return result
    
    @staticmethod
    def __number_of_sentences(text: str) -> int:
        return text.count(".") + text.count("!") + text.count("?")
    
    @staticmethod
    def __number_of_syllables(words: List[str]) -> int:
        count = 0
        vowels = "aeiouy"
        for word in words:
            if not word.isalpha():
                count += 1
                continue
            if word[0] in vowels:
                count += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i - 1] not in vowels:
                    count += 1
                    if word.endswith("e"):
                        count -= 1
            if count == 0:
                count += 1
        return count

    @staticmethod
    def __flesch_reading_ease(avg_sentence_length: float, avg_syllables_per_word: float) -> float:
        return 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

    @staticmethod
    def __flesch_kincaid_grade_level(avg_sentence_length: float, avg_syllables_per_word: float) -> float:
        return 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

    @staticmethod
    def __number_of_numbers(words: List[str]) -> int:
        count = 0
        for word in words:
            if any(char.isdigit() for char in word):
                count += 1
        return count
    
    def __extract_math_expressions(text: str) -> List[str]:
        display_matches = re.findall(r"\$\$(.*?)\$\$", text, re.DOTALL)
        bracket_matches = re.findall(r"\\\[(.*?)\\\]", text, re.DOTALL)
        inline_matches = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text, re.DOTALL)
        return display_matches + bracket_matches + inline_matches
    
    @staticmethod
    def __number_of_operators(math: List[str]) -> int:
        count = 0
        for math_str in math:
            for operator in TEX_OPERATORS:
                count += math_str.count(operator)
        return count
    
    @staticmethod
    def __number_of_variables(math: List[str]) -> int:
        count = 0
        for math_str in math:
            prev_prev = False
            prev = False
            for char in math_str:
                if not char.isalpha():
                    if not prev_prev and prev:
                        count += 1
                    prev_prev = prev
                    prev = False
                else:
                    prev_prev = prev
                    prev = True
            if not prev_prev and prev:
                count += 1
        
        return count


class LingFeatureModel(ModelInterface):
    def __init__(self, normalized_data: NormalizedData = None, bounding_func: Callable[[float], float] = identity, regression_model = None):
        if regression_model is None:
            regression_model = GradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.03,
                n_estimators=500,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=5,
                subsample=0.9,
                max_features=None,
                random_state=67
            )
        else:
            self._model = regression_model

        self._bounding_func = bounding_func

        if normalized_data is not None:
            self.fit(normalized_data.train_data, normalized_data.train_labels)

    def fit(self, data: List[Features], labels: List[float]) -> None:
        X = [LingFeatures(d.description).extract_features() for d in data]
        self._model.fit(X, labels)

    def predict(self, features: Features) -> float:
        X = [LingFeatures(features.description).extract_features()]
        return self._bounding_func(self._model.predict(X)[0])

    def search_fit(self, data: List[Features], labels: List[float], params: Dict[str, List[float]]) -> Dict[str, float]:
        X = [LingFeatures(d.description).extract_features() for d in data]

        search = GridSearchCV(self._model, params, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
        search.fit(X, labels)

        self._model = search.best_estimator_
        return search.best_params_