# Simply neural models for the text experiments
# Alejandro Ciuba, alejandrociuba@pitt.edu
from collections import defaultdict
from gensim.models import KeyedVectors
from nltk import word_tokenize
from torch.utils.data import (DataLoader,
                              TensorDataset, )

import torch

import numpy as np
import torch.nn as nn


class Neural:

    model: nn.Module | None

    lr: float
    epochs: int
    batch_size: int

    device: str

    loss_record = None
    sampler = None

    criterion = None
    optimizer = None
    scheduler = None

    def __init__(self, **config) -> None:

        self.lr = config['lr']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.criterion = nn.BCEWithLogitsLoss() if config['binary'] else nn.CrossEntropyLoss()
        self.scheduler = config['scheduler'] if 'scheduler' in config else None

        self.sampler = config['sampler'] if 'sampler' in config else None
        self.loss_record = defaultdict(list)

        self.model = None

    def fit(self, X, y, step_track: int = 10, verbose: bool = True):

        dataset = TensorDataset(X if isinstance(X, torch.Tensor) else torch.from_numpy(X), y if isinstance(y, torch.Tensor) else torch.from_numpy(y))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True if self.sampler is None else False, num_workers=8, sampler=self.sampler)

        n_total_steps = len(dataloader)

        for epoch in range(self.epochs):
            
            for i, (X, y) in enumerate(dataloader):

                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X)
                loss = self.criterion(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step

                if (i + 1) % (n_total_steps // step_track) == 0:

                    if verbose:
                        print('epoch: %d/%d, step: %d/%d, loss=%.4ff' % (epoch+1, self.epochs, i+1, n_total_steps, loss.item()))

                    self.loss_record['epoch'].append(epoch)
                    self.loss_record['step'].append((i + 1) / n_total_steps)
                    self.loss_record['loss'].append(loss.item())

    def predict(self, X) -> tuple[list, list]:

        dataset = TensorDataset(X if isinstance(X, torch.Tensor) else torch.from_numpy(X), torch.zeros(X.shape[0]))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        prob_layer = nn.Softmax()

        with torch.no_grad():

            preds = None

            for X, y in dataloader:

                X = X.to(self.device)
                y = y.to(self.device)
                outputs = prob_layer(self.model(X))

                _, batch_preds = torch.max(outputs, 1)

                preds = torch.cat([preds, batch_preds]) if preds is not None else batch_preds

            return preds.to(device="cpu").numpy()


class FFNN(Neural):
    """
    FFNN with numpy inputs
    """

    steps: list

    def __init__(self, **config) -> None:

        super().__init__(**config)

        self.steps = config['steps']
        self.model = nn.Sequential(*self.steps)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)


class LSTM(Neural):
    """
    LSTM with numpy inputs
    """

    input_size: int
    hidden_size: int

    num_layers: int
    num_classes: int

    in_linear: list | None = None
    out_linear: list | None = None

    def __init__(self, **config) -> None:

            super().__init__(**config)

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.input_size = config['input_size']
            self.hidden_size = config['hidden_size']

            self.num_layers = config['num_layers']
            self.num_classes = config['num_classes']

            self.in_linear = config['in_linear'] if 'in_linear' in config else None
            self.out_linear = config['out_linear'] if 'out_linear' in config else None

            self.model = _LSTM(self.input_size, self.hidden_size, 
                               self.num_layers, self.num_classes, 
                               self.in_linear, self.out_linear,
                               self.device)
            self.model.to(self.device)

            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)


class _LSTM(nn.Module):

    in_linear: nn.Module | None   # Linear operations before the LSTM
    out_linear: nn.Module | None  # Linear operations after the LSTM

    def __init__(self, input_size, hidden_size, 
                 num_layers, num_classes,
                 in_linear = None, out_linear = None,
                 device = "cpu") -> None:
        
        super(_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        if in_linear:
            self.in_linear = nn.Sequential(*in_linear)
            self.in_linear.to(device)

        if out_linear:
            self.out_linear = nn.Sequential(*out_linear)
            self.out_linear.to(device)

    def forward(self, x):

        out = self.in_linear(x) if self.in_linear else x

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(out, (h0.detach(), c0.detach()))
        out = out[:, -1, :]

        if self.out_linear:
            out = self.out_linear(out)

        return out


class Word2VecEmbeddings:
    """
    Generate the embeddings for a given sentence with padding.
    This is similar to what is done in the original paper, but simpler.
    """

    SOS = "<sos>"
    EOS = "<eos>"
    PAD = "<pad>"
    UNK = "<unk>"
    TOKS = [SOS, EOS, PAD, UNK]

    vocab: set
    total: int
    word2ind: dict
    ind2word: dict
    randvecs: list  # Track which words were not found in your W2V embedding
    w2v: KeyedVectors
    embedding: nn.Embedding

    def __init__(self, sents, w2v) -> None:

        self.vocab = set()
        self.total = 0
        self.word2ind = {}
        self.ind2word = {}
        self.randvecs = []
        self.w2v = w2v

        for tok in self.TOKS:
            self.add_word(tok)

        # Build the vocabulary
        for sent in sents:
            for tok in word_tokenize(sent):
                self.add_word(tok)

        # Get the vectors
        vec_dim = self.w2v["hello"].shape[0]
        arrays = np.zeros((self.total, vec_dim), dtype=np.float32)

        for word in self.vocab:

            if word in self.w2v:
                arrays[self.word2ind[word]] = self.w2v[word]

            else:

                arrays[self.word2ind[word]] = np.random.randn(vec_dim)
                self.randvecs.append(word)

        # Add the padding vector
        arrays[self.word2ind[self.PAD]] = np.zeros(vec_dim, dtype=np.float32)

        arrays = torch.from_numpy(arrays)

        # Put it in the embedding layer
        self.embedding = nn.Embedding.from_pretrained(arrays, freeze=True, 
                                                      padding_idx=self.word2ind[self.PAD])

    def __call__(self, words: list[str]):
        """
        Gotten from the original paper.
        """
        return [self.word2ind[word] if word in self.vocab else self.word2ind[self.UNK] for word in words]

    def add_word(self, word):

        if word not in self.word2ind:

            self.vocab.add(word)
            self.word2ind[word] = self.total
            self.ind2word[self.total] = word
            self.total += 1

    def encode(self, sents, out) -> torch.Tensor:
        """
        Returns the sentences as their |sents|*out*|W2V| word embeddings.
        Includes special token insertion `"hello" -> <sos> hello <eos> <pad>...`
        """

        sents_toks = [word_tokenize(sent) for sent in sents]
        format_toks = [[self.SOS] + sent[:out] + [self.EOS] + ([self.PAD] * (out - len(sent))) \
                       for sent in sents_toks]

        inds = [self(toks) for toks in format_toks]
        inds = torch.tensor(inds, dtype=torch.int32)

        return self.embedding(inds)

if __name__ == "__main__":

    test = ["Hello, there everyone!",
            "I am a test sentence.",
            "This is another one to put in",
            "Smaller sentence here!",
            "And one more for good measure...", ]
    
    WORD2VEC_PATH = "../data/word2vec/GoogleNews-vectors-negative300.bin"   
    W2V: KeyedVectors = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)

    embs = Word2VecEmbeddings(test, W2V)

    print(embs.total, embs.embedding, embs.vocab, embs.randvecs)

    print(embs.encode(test + ["new word beans!"], out=8).shape)
