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


def __is_valid(sample) -> bool:
    try:
        float(sample["level"].split()[1])
        return True
    except (IndexError, ValueError):
        return False


def data_load(split_ratio: float = 0.8, normalize_labels: bool = True, filter: List[ProblemType] = None) -> NormalizedData:
    ds = load_dataset(DATASET_NAME, SUBSET_NAME)

    ds["train"].filter(__is_valid)
    ds["test"].filter(__is_valid)

    if filter is not None:
        def filter_fn(example):
            return TYPE_MAPPING.get(example["type"], None) in filter
        
        ds["train"].filter(filter_fn)
        ds["test"].filter(filter_fn)

    data  = [
        Features(description=sample["problem"], solution=sample["solution"], problem_type=TYPE_MAPPING.get(sample["type"], None))
        for sample in chain(ds["train"], ds["test"])
    ]

    labels = [float(sample["level"].split()[1]) for sample in chain(ds["train"], ds["test"])]

    if normalize_labels:
        labels = [(x - 1) / 4 for x in labels]

    return NormalizedData(data=data, labels=labels, split_ratio=split_ratio)