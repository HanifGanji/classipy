from sentence_transformers import SentenceTransformer
from src.exceptions import NotEnoughWords, NoLabelsDefined
import numpy as np
import pandas as pd


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class WordBasedClassifier:
    def __init__(self, use_cuda=False):
        device = "cuda" if use_cuda else "cpu"
        self.model = SentenceTransformer("all-mpnet-base-v2", device=device)
        self.vector_dict = {}
        self.labels = []

    def __validate_labels(self, label_dict: dict) -> None:
        for label in label_dict.keys():
            if len(label_dict[label]) < 5:
                raise NotEnoughWords(
                    f"Label {label} is expected to have at least 5 words, but got {len(label_dict[label])}!."
                )

    def add_labels(self, label_dict: dict, overwrite: bool = True) -> None:
        self.__validate_labels(label_dict)

        labels = list(label_dict.keys())

        vector_dict = {
            label: vectors
            for label, vectors in zip(
                labels,
                [
                    [self.model.encode(word) for word in label_dict[label]]
                    for label in labels
                ],
            )
        }

        if overwrite:
            self.labels = labels
            self.vector_dict = {}

        self.vector_dict.update(vector_dict)
        self.labels += labels

        del vector_dict

    def predict(self, text: str, get_similarities=False):
        if not self.labels:
            raise NoLabelsDefined(
                "There are no labels defined! make sure that you've called add_labels() before."
            )

        target_vector = self.model.encode(text)
        results = {
            label: np.mean(
                [
                    cosine_similarity(target_vector, vec)
                    for vec in self.vector_dict[label]
                ]
            )
            for label in self.labels
        }

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        if get_similarities:
            return pd.DataFrame(
                zip(
                    [item[0] for item in sorted_results],
                    [item[1] for item in sorted_results],
                ),
                columns=["label", "similarity_with_given_text"],
            ).set_index("label")

        return sorted_results[0][0]
