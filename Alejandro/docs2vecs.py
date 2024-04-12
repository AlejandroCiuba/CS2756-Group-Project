# Train the model for the Doc2Vec vectors
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
from collections import Counter
from gensim.models.doc2vec import (TaggedDocument,
                                   Doc2Vec, )
from gensim.utils import simple_preprocess
from pathlib import Path
from tqdm import tqdm

import argparse

import pandas as pd


PATH = "../data/final-splits.csv"


def main(args: argparse.Namespace):

    data = pd.read_csv(PATH)

    train = [TaggedDocument(simple_preprocess(row.utterance), [i]) for i, row in enumerate(data[data["split"] == "TRAIN"].itertuples())]
    # test = [simple_preprocess(row.utterance) for row in data[data["split"] == "TEST"].itertuples()]

    # Unsure if BLAS is installed
    model = Doc2Vec(vector_size=args.vector, min_count=1)
    model.build_vocab(train)

    model.train(train, total_examples=model.corpus_count, epochs=args.epochs)

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

    file: Path = args.save / f"doc2vec-{args.vector}-{args.epochs}.bin"

    model.save(str(file.absolute()))


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-v",
        "--vector",
        default=75,
        type=int,
        help="Vector size.\n \n",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Epochs.\n \n",
    )

    parser.add_argument(
        "-s",
        "--save",
        default=Path("../data/"),
        type=Path,
        help="File save path.\n \n",
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            prog="docs2vecs.py",
            formatter_class=argparse.RawTextHelpFormatter,
            description="Generate doc2vec vectors based on a pandas dataset.",
            epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
        )

    add_args(parser)
    args = parser.parse_args()

    main(args=args)
