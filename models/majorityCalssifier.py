import numpy as np

class MajorityClassifier:
    def fit(self, y):
        most_frequent = list()
        for column in y.columns:
            _, counts = np.unique(y[column], axis=0, return_counts=True)
            most_frequent.append(np.argmax(counts))
        self.majority_class = most_frequent

    def predict(self, X):

        return np.full(X.shape, self.majority_class)


