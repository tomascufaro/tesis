from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import numpy as np
from sklearn.datasets import make_spd_matrix
from sklearn.model_selection import train_test_split


def max_a_posteriori(x, weights, means, covariances, n_components):
    """
    - x: Es el vector a partir del cual se adapta el modelo
    - weights: es el vector de pesos del UBM tiene forma [n_components, n_features]
    - means: es el vector de medias del UBM, tiene forma [n_components,]
    - covariances: es la matriz de covarianzas del UBM, tiene forma [n_components, n_features, n_features]
    """
    R = 16  # factor de relevancia
    pr = np.zeros([n_components, len(x)])  #  Wi * Pi(xt) / Sum(Wj . Pj(xt))
    n_features = len(x[0])
    for i in range(n_components):
        for n, xt in enumerate(x):
            pr[i, n] = weights[i] * multivariate_normal.pdf(
                xt,
                means[i],
                covariances[
                    i
                ],  #  Se calcula la probabilidad de que xt pertenezca a una normal con media means[i] y varianza covariances[i]
            )

    for n in range(len(x)):
        pr[:, n] = pr[:, n] / sum(pr[:, n])  #  Dimension [1, n_components]

    mid_weights = np.zeros(n_components)  #  ni
    mid_mean = np.zeros([n_components, len(x[0])])  #  Ei(x)
    mid_covariances = np.zeros([n_components, len(x[0]), len(x[0])])
    final_weights = np.zeros(n_components)
    final_mean = np.zeros([n_components, len(x[0])])  #  Ei(x)
    final_covariances = np.zeros([n_components, len(x[0]), len(x[0])])
    alpha = np.zeros(n_components)
    # pr[i, n] es escalar.

    for i in range(n_components):
        for n, xt in enumerate(x):
            # xt es un vector con longitud igual a la cantidad de features de entrada.
            mid_weights[i] += pr[i, n]
            mid_mean[i, :] += pr[i, n] * xt
            xt = np.expand_dims(xt, -1)
            mid_covariances[i, :, :] += pr[i, n] * (np.matmul(xt, xt.T))

        mid_mean[i, :] = mid_mean[i] / mid_weights[i]

        mid_covariances[i, :, :] = mid_covariances[i, :, :] / mid_weights[i]

        alpha[i] = mid_weights[i] / (mid_weights[i] * R)

        final_weights[i] = (alpha[i] * mid_weights[i] / len(x)) + (
            (1 - alpha[i]) * weights[i]
        )

        final_mean[i, :] = alpha[i] * mid_mean[i, :] + (1 - alpha[i]) * means[i, :]

        squared_mid_mean = np.matmul(
            np.expand_dims(mid_mean[i, :], -1), np.expand_dims(mid_mean[i, :], -1).T
        )

        squared_final_mean = np.matmul(
            np.expand_dims(final_mean[i, :], -1), np.expand_dims(final_mean[i, :], -1).T
        )

        final_covariances[i, :, :] = (
            alpha[i] * mid_covariances[i, :, :]
            + (1 - alpha[i]) * (covariances[i, :, :] + squared_mid_mean)
            - squared_final_mean
        )

    final_weights = final_weights / sum(final_weights)

    return final_weights, final_mean, final_covariances


class Emotion_Gmm(object):
    def __init__(self, n_components):
        self.ubm = GaussianMixture(n_components=n_components)
        self.ubm_trained = False
        self.n_components = n_components
        self.model_trained = False

    def fit(self, x):
        if self.ubm_trained:
            self.weights_, self.means_, self.covariances_ = max_a_posteriori(
                x,
                self.ubm.weights_,
                self.ubm.means_,
                self.ubm.covariances_,
                self.n_components,
            )
            self.model_trained = True
        else:
            print("El UBM no se entrenó")

    def fit_UBM(self, x):
        self.ubm.fit(x)
        self.ubm_trained = True

    def predict(self, x, c="all"):
        """
        x = vector a predecir
        c = cantidad de gaussianas que se consideran para calcular el log-likelihood (Mayor a 0), si no se pone ningun valor se usan todas.
        """
        if self.model_trained:
            p = 0
            p_ubm = 0
            for i in range(self.n_components):
                # print(self.covariances_[i].size)
                p += self.weights_[i] * multivariate_normal.pdf(
                    x,
                    self.means_[i],
                    self.covariances_[
                        i, :, :
                    ],  #  Se calcula la probabilidad de que xt pertenezca a una normal con media means[i] y varianza covariances[i]
                )
                p_ubm += self.ubm.weights_[i] * multivariate_normal.pdf(
                    x,
                    self.ubm.means_[i],
                    self.ubm.covariances_[
                        i
                    ],  #  Se calcula la probabilidad de que xt pertenezca a una normal con media means[i] y varianza covariances[i]
                )
                log_likelihood = np.log(p / p_ubm)
        return log_likelihood


# Si todo esta bien, se deberia poder entrenar un UBM a partir de GaussianMixture y luego a partir del UBM entrenar un GMM para detectar discusiones usando Emotion_Gmm
# No se como va a funcionar cuando los features no sean escalares!!!

if __name__ == "__main__":

    n_components = 6
    directory = "./dataset3_mp3.npy"
    dataset = np.load(directory, allow_pickle=True)
    x = dataset[()]["x"]
    Y = dataset[()]["y"]
    names = dataset[()]["names"]
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(x), Y, test_size=0.1, random_state=9
    )

    print(x_train.shape)
    print("------")
    print(x_train[0].shape)
    print("----===")
    print(x_train[0])
    # x_positive = []
    # x_background = []
    # for x, y in zip(x_train, y_train):
    #     if y == 1:
    #         x_positive.append(x)
    #     else:
    #         x_background.append(x)
    # # x_positive = np.array(x_positive)
    # # x_background = np.array(x_background)

    # print(x_positive.shape)
    # print("------")
    # print(x_positive[0].shape)
    # print("------")
    # print(x_positive[0])
    # anger_detector = Emotion_Gmm(n_components)
    # anger_detector.fit_UBM(x_background)
    # anger_detector.fit(x_positive)

    # pred = anger_detector.predict(x_train[0])
    # print(pred, y_test[0])