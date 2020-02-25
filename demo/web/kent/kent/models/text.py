import numpy as np

from collections import defaultdict
from itertools import chain, islice, tee
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

from . import Model

UNKNOWN = '<UNKNOWN>'

def _ngram(iterable, n=2, pad='<PAD>', delimeter='/'):
    padded = chain(
        [pad] * (n - 1),
        iterable,
        [pad] * (n - 1),
    )

    iterables = [
        islice(padded_i, i, None)
        for i, padded_i in enumerate(tee(padded, n))
    ]
    yield from map(delimeter.join, zip(*iterables))


class TextVectorizer(Model):
    pass


class BagOfWordsTextVectorizer(TextVectorizer):
    def __init__(self, ngram_range=range(1, 2 + 1), vocabulary_size=None):
        self.ngram_range = ngram_range
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        count_by_term = defaultdict(int)
        for term in self._ngrams(X):
            count_by_term[term] += 1

        n = (self.vocabulary_size - 1) if self.vocabulary_size else -1
        top_n = sorted(count_by_term.items(), key=lambda item: item[1], reverse=True)[:n]
        terms = [UNKNOWN] + [term for term, count in top_n]

        self.vocabulary_size = len(terms)
        self.vocabulary = {
            word: index for index, word in enumerate(terms)
        }

        return self

    def transform(self, X):
        X_transformed = np.zeros((len(X), len(self.vocabulary)), dtype=int)
        for x, x_transformed in zip(X, X_transformed):
            for term in self._ngrams([x]):
                x_transformed[self.vocabulary.get(term, 0)] = 1

        return X_transformed

    def _ngrams(self, X):
        for n in self.ngram_range:
            for x in X:
                yield from _ngram(x, n)


class TextClassifier(Model):
    def __init__(self):
        self.text_vectorizer = BagOfWordsTextVectorizer()
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        X_vectorized = self.text_vectorizer.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X_vectorized, y_encoded)

    def predict(self, X):
        X_vectorized = self.text_vectorizer.transform(X)
        y_predicted = self.model.predict_proba(X_vectorized)
        
        return list(zip(self.label_encoder.classes_, y_predicted[0]))


class MLPTextClassifier(TextClassifier):
    def __init__(self):  # configurations
        super().__init__()
        self.model = MLPClassifier(verbose=True)


class RandomTextClassifier(Model):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.labels = list(set(y))

    def predict(self, X):
        import random
        return [
            (random.choice(self.labels), random.uniform(0.5, 1.0))
            for _ in X
        ]
