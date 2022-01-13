import random
from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class Scaler(ABC):
    @abstractmethod
    def scale(self, x, y):
        pass


class MinMax_Scaler(Scaler):
    def scale(self, x, y):
        return MinMaxScaler().fit_transform(x, y)


class Standard_Scaler(Scaler):
    def scale(self, x, y):
        return StandardScaler().fit_transform(x, y)


class Preprocessor:
    def __init__(self, scaler: Scaler, seed: int):
        self.seed = seed
        self.scaler = scaler

    def scale(self, x, y):

        return self.scaler.scale(x, y)

    def downsample(
        self,
        x,
        y,
        condition,
        rate,
    ):
        random.seed(self.seed)
        l = np.count_nonzero(np.array(y) == condition)
        n = int(l * rate)
        indexes = list(np.where(np.array(y) == condition)[0])
        indexes = random.sample(indexes, n)
        x_downsampled = np.delete(np.array(x), indexes, 0)
        y_downsampled = np.delete(y, indexes, 0)

        return x_downsampled, y_downsampled

    def shuffle(self, x, y):
        x, y = shuffle(x, y, random_state=self.seed)
        return x, y

    def delete_features(self):
        pass
