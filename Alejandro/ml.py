# Class which will train and run non-neural models
from sklearn.feature_extraction.text import (TfidfVectorizer,
                                             CountVectorizer, )
from sklearn.metrics import (accuracy_score,
                             confusion_matrix, )
from sklearn.naive_bayes import (MultinomialNB,
                                 ComplementNB, )
from sklearn.pipeline import (make_pipeline,
                              Pipeline, )
from sklearn.svm import SVC

import pandas as pd

class MLModel:

    name: str
    data: pd.DataFrame
    sents: str
    labels: str
    pipeline: Pipeline

    def __init__(self, name: str, data: pd.DataFrame, sents: str, labels: str):

        self.name = name
        self.data = data
        self.sents = sents
        self.labels = labels

    @property
    def train_samples(self) -> tuple[list[str], list[str]]:
        return self.data[self.data["split"] == "TRAIN"][self.sents].to_list(), self.data[self.data["split"] == "TRAIN"][self.labels].to_list()

    @property
    def valid_samples(self) -> tuple[list[str], list[str]]:
        return self.data[self.data["split"] == "VALID"][self.sents].to_list(), self.data[self.data["split"] == "VALID"][self.labels].to_list()

    @property
    def test_samples(self) -> tuple[list[str], list[str]]:
        return self.data[self.data["split"] == "TEST"][self.sents].to_list(), self.data[self.data["split"] == "TEST"][self.labels].to_list()

    def steps(self, *args):
        self.pipeline = make_pipeline(*args)

    def train(self):
        self.pipeline.fit(self.train_samples[0], self.train_samples[1])

    def test(self) -> tuple[int, int]:
        return self.pipeline.predict(self.test_samples[0]), self.test_samples[1]

    def full_run(self):

        self.train()
        pred_y, y = self.test()

        print(f"{self.name}:")
        print("Accuracy:", accuracy_score(y, pred_y))

if __name__ == "__main__":

    data = pd.read_csv("../data/small-splits.csv")

    model = MLModel(name="SVC/COUNT", data=data, sents="utterance", labels="emotion")
    model.steps(CountVectorizer(stop_words="english", max_features=1_000), SVC(C=10))
    model.full_run()

    model = MLModel(name="SVC/TF-IDF", data=data, sents="utterance", labels="emotion")
    model.steps(TfidfVectorizer(stop_words="english", max_features=1_000), SVC(C=10))
    model.full_run()

    model = MLModel(name="MB/COUNT", data=data, sents="utterance", labels="emotion")
    model.steps(CountVectorizer(stop_words="english", max_features=1_000), MultinomialNB())
    model.full_run()

    model = MLModel(name="MB/TF-IDF", data=data, sents="utterance", labels="emotion")
    model.steps(TfidfVectorizer(stop_words="english", max_features=1_000), MultinomialNB())
    model.full_run()

    model = MLModel(name="CB/COUNT", data=data, sents="utterance", labels="emotion")
    model.steps(CountVectorizer(stop_words="english", max_features=1_000), ComplementNB())
    model.full_run()

    model = MLModel(name="CB/TF-IDF", data=data, sents="utterance", labels="emotion")
    model.steps(TfidfVectorizer(stop_words="english", max_features=1_000), ComplementNB())
    model.full_run()
