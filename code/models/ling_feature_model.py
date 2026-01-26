from typing import Dict, List, Optional, Callable
import re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from textstat import textstat

from models.model_interface import ModelInterface
from normalized_data import *
from utils import *


TEX_OPERATORS = ["\\frac", "\\dfrac", "\\sum", "\\lim", "\\sin", "\\cos", "\\tan", "\\log", "\\ln", "\\sqrt", "\\leq", "\\le", "\\geq", "\\ge", "\\neq", "+", "-", "*", "/", "=", "<", ">", "^"]


class LingFeatures:
    def __init__(self, input_text: str):
        text = input_text.lower()
        text_no_tex = normalize_whitespaces(remove_tex(text))
        math = LingFeatures.__extract_math_expressions(text)

        self.sentence_count = textstat.sentence_count(text_no_tex)
        self.lexicon_count = textstat.lexicon_count(text_no_tex, removepunct=True)
        self.syllable_count = textstat.syllable_count(text_no_tex)

        self.flesch_reading_ease = textstat.flesch_reading_ease(text_no_tex)
        self.flesch_kincaid_grade_level = textstat.flesch_kincaid_grade(text_no_tex)
        self.dale_chall_readability_score = textstat.dale_chall_readability_score(text_no_tex)
        self.difficulty_words = textstat.difficult_words(text_no_tex)

        self.math_expression_count = len(math)
        self.numbers_count = LingFeatures.__numbers_count(text.split())
        self.operators_count = LingFeatures.__operators_count(math)
        self.variables_count = LingFeatures.__variables_count(math)

    def feature_vector(self) -> List[float]:
        return [
            self.sentence_count,
            self.lexicon_count,
            self.syllable_count,
            self.flesch_reading_ease,
            self.flesch_kincaid_grade_level,
            self.dale_chall_readability_score,
            self.difficulty_words,
            self.math_expression_count,
            self.numbers_count,
            self.operators_count,
            self.variables_count
        ]
    
    @staticmethod
    def feature_names() -> List[str]:
        return [
            "sentence_count",
            "lexicon_count",
            "syllable_count",
            "flesch_reading_ease",
            "flesch_kincaid_grade_level",
            "dale_chall_readability_score",
            "difficulty_words",
            "math_expression_count",
            "numbers_count",
            "operators_count",
            "variables_count"
        ]
    
    def __extract_math_expressions(text: str) -> List[str]:
        display_matches = re.findall(r"\$\$(.*?)\$\$", text, re.DOTALL)
        bracket_matches = re.findall(r"\\\[(.*?)\\\]", text, re.DOTALL)
        inline_matches = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text, re.DOTALL)
        return display_matches + bracket_matches + inline_matches

    @staticmethod
    def __numbers_count(words: List[str]) -> int:
        count = 0
        for word in words:
            if any(char.isdigit() for char in word):
                count += 1
        return count
    
    @staticmethod
    def __operators_count(math: List[str]) -> int:
        count = 0
        for math_str in math:
            for operator in TEX_OPERATORS:
                count += math_str.count(operator)
        return count
    
    @staticmethod
    def __variables_count(math: List[str]) -> int:
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
    def __init__(self,
        regressor = GradientBoostingRegressor(
            loss="squared_error",
            max_features=None,
            random_state=67,
            learning_rate=0.01,
            n_estimators=500,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=3,
            subsample=0.6
        ),
        bounding_func: Callable[[float], float] = clip
    ):
        
        self._model = regressor
        self._bounding_func = bounding_func

    def fit(self, data: List[Features], labels: List[float]) -> None:
        X = [LingFeatures(d.description).feature_vector() for d in data]
        self._model.fit(X, labels)

    def predict(self, features: Features) -> float:
        X = [LingFeatures(features.description).feature_vector()]
        return self._bounding_func(self._model.predict(X)[0])

    def search_fit(self, data: List[Features], labels: List[float], params: Dict[str, List[float]]) -> Dict[str, float]:
        X = [LingFeatures(d.description).feature_vector() for d in data]

        search = GridSearchCV(self._model, params, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
        search.fit(X, labels)

        self._model = search.best_estimator_
        return search.best_params_
    
    def feature_importances(self) -> Optional[Dict[str, float]]:
        if hasattr(self._model, "feature_importances_"):
            return dict(zip(LingFeatures.feature_names(), self._model.feature_importances_.tolist()))
        return None