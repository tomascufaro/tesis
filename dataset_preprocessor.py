<<<<<<< HEAD
import random
from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import scipy as sp
from scipy import stats
from sklearn.utils import shuffle
from sklearn.covariance import MinCovDet


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
    def __init__(self, scaler: Scaler = None, seed: int = None):
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

    def __get_outliers(self, x):
        # Minimum covariance determinant
        rng = np.random.RandomState(0)
        real_cov = np.cov(x.T)
        X = rng.multivariate_normal(mean=np.mean(x, axis=0), cov=real_cov, size=506)
        cov = MinCovDet(random_state=0).fit(X)
        mcd = cov.covariance_  # robust covariance metric
        robust_mean = cov.location_  # robust mean
        inv_covmat = sp.linalg.inv(mcd)  # inverse covariance metric

        # Robust M-Distance
        x_minus_mu = x - robust_mean
        left_term = np.dot(x_minus_mu, inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
        md = np.sqrt(mahal.diagonal())

        # Flag as outlier
        outlier = []
        C = np.sqrt(
            stats.chi2.ppf((1 - 0.001), df=19)
        )  # degrees of freedom = number of variables
        for index, value in enumerate(md):
            if value > C:
                outlier.append(index)
            else:
                continue
        return outlier, md

    def get_outliers_id(self, db):

        (
            dataset_x,
            dataset_y,
            _,
            _,
            _,
            _,
        ) = db.get_datasets()

        positive_indexes = np.where(np.array(dataset_y) == 1)
        negative_indexes = np.where(np.array(dataset_y) == 0)
        negative_cases = dataset_x[negative_indexes]
        positive_cases = dataset_x[positive_indexes]

        positive_outliers = self.__get_outliers(positive_cases)[0]
        negative_outliers = self.__get_outliers(negative_cases)[0]
        outlier_indexes = np.concatenate((positive_outliers, negative_outliers), axis=0)

        ids = list(db.dataset_no_aug.iloc[outlier_indexes]["_id"])
        ids = map(
            lambda x: x.replace("iemocap_", "IEMOCAP_mp3/").replace("mp3_", "mp3"), ids
        )

        return list(ids)
=======
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
    def __init__(self, scaler: Scaler, seed: int, db=None):
        self.seed = seed
        self.scaler = scaler
        self.db = db

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

    def get_outliers(self):
        if self.db == None:
            raise ("Error: No Database")
        else:
            pass
>>>>>>> 7a644d3799ae563dbb17b7b353102d0f86daf149
