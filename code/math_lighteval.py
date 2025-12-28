from datasets import load_dataset
from normalized_data import *

DATASET_NAME = "DigitalLearningGmbH/MATH-lighteval"

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


def data_load(normalize_labels: bool = False, filter: List[ProblemType] = None) -> NormalizedData:
    ds = load_dataset(path=DATASET_NAME)

    ds["train"] = ds["train"].filter(__is_valid)
    ds["test"] = ds["test"].filter(__is_valid)

    if filter is not None:
        def filter_fn(example):
            return TYPE_MAPPING.get(example["type"], None) in filter
        ds["train"] = ds["train"].filter(filter_fn)
        ds["test"] = ds["test"].filter(filter_fn)

    train_data = [
        Features(description=sample["problem"], solution=sample["solution"], problem_type=TYPE_MAPPING.get(sample["type"], None))
        for sample in ds["train"]
    ]

    test_data = [
        Features(description=sample["problem"], solution=sample["solution"], problem_type=TYPE_MAPPING.get(sample["type"], None))
        for sample in ds["test"]
    ]

    train_labels = [float(sample["level"].split()[1]) for sample in ds["train"]]
    test_labels = [float(sample["level"].split()[1]) for sample in ds["test"]]

    if normalize_labels:
        train_labels = [x / 5.0 for x in train_labels]
        test_labels = [x / 5.0 for x in test_labels]

    return NormalizedData(train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)