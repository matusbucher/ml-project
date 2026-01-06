from typing import List
from datasets import load_dataset
from itertools import chain

from normalized_data import *

DATASET_NAME = "DigitalLearningGmbH/MATH-lighteval"
SUBSET_NAME = "default"

TYPE_MAPPING = {
    "Algebra": ProblemType.ALGEBRA,
    "Intermediate Algebra": ProblemType.INTERMEDIATE_ALGEBRA,
    "Prealgebra": ProblemType.PREALGEBRA,
    "Geometry": ProblemType.GEOMETRY,
    "Number Theory": ProblemType.NUMBER_THEORY,
    "Counting & Probability": ProblemType.COUNTING_AND_PROBABILITY,
    "Precalculus": ProblemType.PRECALCULUS,
}


async def __is_valid(sample) -> bool:
    try:
        float(sample["level"].split()[1])
        return True
    except (IndexError, ValueError):
        return False


def data_load(test_ratio: float = 0.2, normalize_labels: bool = True, filter_type: List[ProblemType] = None) -> NormalizedData:
    ds = load_dataset(DATASET_NAME, SUBSET_NAME)

    ds["train"] = ds["train"].filter(__is_valid)
    ds["test"] = ds["test"].filter(__is_valid)

    if filter_type is not None:
        async def filter_fn(sample):
            return TYPE_MAPPING.get(sample["type"], None) in filter_type
        ds["train"] = ds["train"].filter(filter_fn)
        ds["test"] = ds["test"].filter(filter_fn)

    data  = [
        Features(description=sample["problem"], solution=sample["solution"], problem_type=TYPE_MAPPING.get(sample["type"], None))
        for sample in chain(ds["train"], ds["test"])
    ]

    labels = [float(sample["level"].split()[1]) for sample in chain(ds["train"], ds["test"])]

    if normalize_labels:
        labels = [(x - 1) / 4 for x in labels]

    return NormalizedData(data=data, labels=labels, test_ratio=test_ratio)