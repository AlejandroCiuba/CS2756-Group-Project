# Train the model for the Doc2Vec vectors
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
from collections import Counter
from gensim.models.doc2vec import (TaggedDocument,
                                   Doc2Vec, )
from gensim.utils import simple_preprocess

import gensim

import pandas as pd


PATH = "../data/final-splits.csv"
EPOCHS = 50

def main():

    data = pd.read_csv(PATH)

    train = [TaggedDocument(simple_preprocess(row.utterance), [i]) for i, row in enumerate(data[data["split"] == "TRAIN"].itertuples())]
    test = [simple_preprocess(row.utterance) for row in data[data["split"] == "TEST"].itertuples()]

    # Unsure if BLAS is installed
    model = Doc2Vec(vector_size=50, min_count=2)
    model.build_vocab(train)

    model.train(train, total_examples=model.corpus_count, epochs=EPOCHS)

    ranks = []
    second_ranks = []
    for doc_id in range(len(train)):
        inferred_vector = model.infer_vector(train[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

    second_ranks.append(sims[1])
    counter = Counter(ranks)
    print(counter)

    model.save("../data/doc2vec.bin")

if __name__ == "__main__":
    main()
