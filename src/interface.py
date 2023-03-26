from src.word_base_classifier import WordBasedClassifier
from src.exceptions import InvalidClassificationMethod


valid_methods = ["word_based"]


def get_classifier(method="word_based", **kwargs):
    if method == "word_based":
        return WordBasedClassifier(**kwargs)
    else:
        raise InvalidClassificationMethod(
            f"{method} is not a valid classification method. Supported methods are:\n{valid_methods}."
        )
