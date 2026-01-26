from typing import Dict, List, Optional
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from models.model_interface import ModelInterface
from normalized_data import *
from utils import *


BERT_BASE_MODEL_NAME = "google-bert/bert-base-uncased"
MATH_BERT_MODEL_NAME = "tbs17/MathBERT"

DEFAULT_MAX_LENGTH = 256
DEFAULT_EPOCHS = 4
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_HIDDEN_DROPOUT_PROB = 0.2
DEFAULT_ATTENTION_PROBS_DROPOUT_PROB = 0.2
DEFAULT_CLIP_FUNC = clip
DEFAULT_OUTPUT_DIR = "./tmp"


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class TransformerRegressor(ModelInterface):
    def __init__(self,
        model_name: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        validation_split: float = DEFAULT_VALIDATION_SPLIT,
        hidden_dropout_prob: float = DEFAULT_HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob: float = DEFAULT_ATTENTION_PROBS_DROPOUT_PROB,
        bounding_func: Callable[[float], float] = DEFAULT_CLIP_FUNC,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        resume: bool = False
    ):
        self._max_length = max_length
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._validation_split = validation_split
        self._bounding_func = bounding_func
        self._output_dir = output_dir
        self._resume = resume

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        ).to(self._device)

    def fit(self, data: List[Features], labels: List[float]) -> None:
        texts = [d.description for d in data]
        train_texts, validation_texts, train_labels, validation_labels = train_test_split(
            texts, labels, test_size=self._validation_split, random_state=67
        ) if self._validation_split > 0.0 else (texts, None, labels, None)

        print(f"Training samples: {len(train_texts)}")
        if validation_texts is not None:
            print(f"Validation samples: {len(validation_texts)}")

        train_torch_labels = torch.tensor(train_labels, dtype=torch.float32)
        train_encodings = self.__tokenize(train_texts)
        train_dataset = TorchDataset(train_encodings, train_torch_labels)

        validation_dataset = None
        if validation_texts is not None and len(validation_texts) > 0:
            validation_torch_labels = torch.tensor(validation_labels, dtype=torch.float32)
            validation_encodings = self.__tokenize(validation_texts)
            validation_dataset = TorchDataset(validation_encodings, validation_torch_labels)
        use_validation = validation_dataset is not None

        training_args = TrainingArguments(
            output_dir=self._output_dir,
            num_train_epochs=self._epochs,
            per_device_train_batch_size=self._batch_size,
            learning_rate=self._learning_rate,
            weight_decay=self._weight_decay,
            eval_strategy="epoch" if use_validation else "no",
            metric_for_best_model="eval_loss" if use_validation else "loss",
            load_best_model_at_end=use_validation,
            logging_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none"
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self._tokenizer,
        )

        trainer.train(resume_from_checkpoint=self._resume)

    def predict(self, features: Features) -> float:
        self._model.eval()

        inputs = self.__tokenize([features.description])
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model(**inputs)
            value = output.logits.squeeze().item()

        return self._bounding_func(value)
    
    def __tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self._tokenizer(
            texts,
            max_length=self._max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )


class BertBaseModel(TransformerRegressor):
    def __init__(
        self,
        max_length: int = DEFAULT_MAX_LENGTH,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        validation_split: float = DEFAULT_VALIDATION_SPLIT,
        hidden_dropout_prob: float = DEFAULT_HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob: float = DEFAULT_ATTENTION_PROBS_DROPOUT_PROB,
        bounding_func: Callable[[float], float] = DEFAULT_CLIP_FUNC,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        resume: bool = False
    ):
        super().__init__(
            model_name=BERT_BASE_MODEL_NAME,
            max_length=max_length,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_split=validation_split,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            bounding_func=bounding_func,
            output_dir=output_dir,
            resume=resume
        )


class MathBertModel(TransformerRegressor):
    def __init__(
        self,
        max_length: int = DEFAULT_MAX_LENGTH,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        validation_split: float = DEFAULT_VALIDATION_SPLIT,
        hidden_dropout_prob: float = DEFAULT_HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob: float = DEFAULT_ATTENTION_PROBS_DROPOUT_PROB,
        bounding_func: Callable[[float], float] = DEFAULT_CLIP_FUNC,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        resume: bool = False
    ):
        super().__init__(
            model_name=MATH_BERT_MODEL_NAME,
            max_length=max_length,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_split=validation_split,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            bounding_func=bounding_func,
            output_dir=output_dir,
            resume=resume
        )