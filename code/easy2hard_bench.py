from datasets import load_dataset
from itertools import chain
from normalized_data import *

DATASET_NAME = "furonghuang-lab/Easy2Hard-Bench"
SUBSET_NAME = "E2H-AMC"


def data_load(split_ratio: float = 0.8) -> NormalizedData:
    ds = load_dataset(DATASET_NAME, SUBSET_NAME)

    data = [
        Features(description=sample["problem"], solution=sample["solution"], problem_type=None)
        for sample in chain(ds["train"], ds["eval"])
    ]

    labels = [float(sample["rating"]) for sample in chain(ds["train"], ds["eval"])]

    return NormalizedData(data=data, labels=labels, split_ratio=split_ratio)