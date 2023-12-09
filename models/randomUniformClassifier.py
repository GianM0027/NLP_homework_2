import numpy as np

class RandomUniformClassifier:
    def __init__(self, n_classes: int):
        self.num_classes = n_classes

    def predict(self, x):
        num_samples = x.shape[0]
        return np.random.randint(0,2,(num_samples, self.num_classes))
    



