from typing import List
from datasets import load_dataset
from itertools import chain

from normalized_data import *

DATASET_NAME = "DigitalLearningGmbH/MATH-lighteval"
SUBSET_NAME = "default"


async def __is_valid(sample) -> bool:
    try:
        float(sample["level"].split()[1])
        return True
    except (IndexError, ValueError):
        return False
    
def __normalize(label: int) -> float:
    return 0.1 + 0.2 * (label - 1)

def data_load(test_ratio: float, normalize_labels: bool = True) -> NormalizedData:
    ds = load_dataset(DATASET_NAME, SUBSET_NAME)

    ds["train"] = ds["train"].filter(__is_valid)
    ds["test"] = ds["test"].filter(__is_valid)

    data  = [
        Features(description=sample["problem"], solution=sample["solution"])
        for sample in chain(ds["train"], ds["test"])
    ]

    labels = [float(sample["level"].split()[1]) for sample in chain(ds["train"], ds["test"])]

    if normalize_labels:
        labels = [__normalize(x) for x in labels]

    return NormalizedData(data=data, labels=labels, test_ratio=test_ratio)