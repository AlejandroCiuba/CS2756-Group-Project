# Class which will train and run models
from sklearn.svm import SVC
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Experiment:

    name: str
    comments: str
    model: Any

    def __init__(self, name: str, data: pd.DataFrame, X: str = "utterance", y: str = "emotion", comments: str = "", model: Any = None):
        """
        Parameters
        ---

        `name`: `str`
            Name of the experiment.

        `data`: `pd.DataFrame`
            DataFrame which contains the features, labels and splits (`"TRAIN"`, `"VALID"`, `"TEST"`).

        `X`: `str`
            Name of the features column.

        `y`: `str`
            Name of the output column.

        `comments`: `str`
            Comments on the experiment.

        `model`: `Any`
            Model to run the experiment.
        """

        self.name = name
        self.comments = comments

        self.train_X, self.train_y = data[data.split == "TRAIN"][X], data[data.split == "TRAIN"][y]
        self.test_X, self.test_y = data[data.split == "TEST"][X], data[data.split == "TEST"][y]
        self.valid_X, self.valid_y = data[data.split == "VALID"][X], data[data.split == "VALID"][y]

        self.model = model

    def transform(self, operation, **kwargs):

        self.train_X, self.train_y = operation(self.train_X, self.train_y, subset="train", **kwargs)
        self.test_X, self.test_y = operation(self.test_X, self.test_y, subset="test", **kwargs)
        self.valid_X, self.valid_y = operation(self.valid_X, self.valid_y, subset="valid", **kwargs)

    def train(self, **kwargs):
        self.model.fit(self.train_X, self.train_y, **kwargs)

    def test(self, subset: str = "test") -> tuple[int, int]:

        if subset == "train":
            return self.train_y, self.model.predict(self.train_X)
        elif subset == "valid":
            return self.valid_y, self.model.predict(self.valid_X)

        return self.test_y, self.model.predict(self.test_X)

    def full_run(self, metrics: dict[str, tuple[Any, dict[str, Any]]], post = None, **kwargs):
        """
        Parameters
        ---

        `metrics`: `dict[str, tuple[Any, dict[str, Any]]]`
            A list of metrics which take the `y_true` and `y_pred`.
            ```
            {"accuracy": (accuracy_score, {}), }
            ```

        `**kwargs`: `Any
            Any arguments for the `Experiment.train` method.
            Also a `"post"` method for y_true and y_pred if applicable.
            Useful if you need to have your outputs match for certain metrics.

        Returns
        ---

        A `dict` containing the metric name and its output.
        """

        # Train the model
        self.train(**kwargs)

        # Get test results
        y_true, y_pred = self.test()

        if post:
            y_true, y_pred = post(y_true, y_pred)

        return {name: metrics[name][0](y_true, y_pred, **metrics[name][1]) for name in metrics}


def plot_confusion_matrix(cm, labels):
    """
    Plots the confusion matrix (thanks to Dr. Na-Rae Han).
    """
    sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True, cmap="Reds", 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()


def vectorizer(X, y, subset, vect) -> tuple[Any, Any]:
    """
    Take an sklearn Vectorizer and use it for experiments.
    """
    if subset == "train":
        return vect.fit_transform(X), y

    return vect.transform(X), y


if __name__ == "__main__":

    # Example usage
    from sklearn.feature_extraction.text import (CountVectorizer, )
    from sklearn.metrics import (accuracy_score,
                                confusion_matrix, )

    data = pd.read_csv("../data/small-splits.csv")

    metrics = {"accuracy": (accuracy_score, {}),
               "conf_mat": (confusion_matrix, {})}

    experiment = Experiment(name="SVC/COUNT", data=data, X="utterance", y="emotion", model=SVC(C=1))
    experiment.transform(vectorizer, vect=CountVectorizer(stop_words="english", max_features=1_000))
    print(experiment.full_run(metrics=metrics))
