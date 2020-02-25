import pickle


class Model:
    def save(self, filename):
        with open(filename, 'wb') as output_file:
            pickle.dump(self, output_file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as input_file:
            return pickle.load(input_file)

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X):
        return self.fit(X, None).transform(X)
