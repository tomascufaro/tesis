import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


class Metrics:
    accuracy = None
    precision = None
    specifity = None
    recall = None
    true_negative_rate = None
    f_score = None

    def set_metrics(
        self, accuracy, precision, specifity, recall, true_negative_rate, f_score
    ):
        self.accuracy = accuracy
        self.precision = precision
        self.specifity = specifity
        self.recall = recall
        self.true_negative_rate = true_negative_rate
        self.f_score = f_score
        self.metrics = [
            accuracy,
            precision,
            specifity,
            recall,
            true_negative_rate,
            f_score,
        ]

    def __str__(self):
        text = textwrap.dedent(
            f"""\
            Exactitud: {self.accuracy}
            Precisión: {self.precision}
            Especificidad: {self.specifity}
            Sensibilidad: {self.recall}
            true_negative_rate: {self.true_negative_rate}
            F1 - Score: {self.f_score}"""
        )
        return text


class Evaluator:
    def __predict(self, model, x, y, threshold=0.5):
        y_pred = model.predict(x)
        y_predicted = []

        for pred in y_pred:
            if pred > threshold:
                y_predicted.append(1)
            else:
                y_predicted.append(0)

        return y_predicted

    def __conf_matrix(self, y, y_predicted):

        tn, fp, fn, tp = confusion_matrix(y, y_predicted).ravel()

        return tn, fp, fn, tp

    def __calc_f_score(self, precision, recall, beta):

        beta_square = beta ** 2
        f = (
            (1 + beta_square)
            * precision
            * recall
            / ((beta_square * precision) + recall)
        )

        return f

    def __calc_metrics(self, tn, fp, fn, tp, beta):

        accuracy = round(
            (tp + tn) / (fn + fp + tp + tn), 2
        )  # Exactitud, porcentaje de predicciones correctas
        precision = round(
            tp / (fp + tp), 2
        )  # Precisión, porcentaje de predicciones positivas correctas
        specifity = round(
            tn / (tn + fn), 2
        )  # Especificidad, porcentaje de casos negativos detectados correctamente
        recall = round(
            tp / (tp + fn), 2
        )  # sensibilidad, porcentaje de casos positivos detectados
        true_negative_rate = round(tn / (tn + fp), 2)
        f_score = self.__calc_f_score(precision, recall, beta)
        m = Metrics()
        m.set_metrics(
            accuracy, precision, specifity, recall, true_negative_rate, f_score
        )
        return m

    def plot_roc_curve(self, y, y_predicted):

        nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y, y_predicted)
        auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
        _, ax1 = plt.subplots(1, 1, figsize=(20, 15))
        ax1.plot(nn_fpr_keras, nn_tpr_keras, label="ROC (auc = %0.3f)" % auc_keras)
        ax1.set_title("ROC", fontsize=22)
        ax1.set_xlabel("False positive rate", fontsize=18)
        ax1.set_ylabel("True positive rate", fontsize=18)

        ticks = np.arange(0, 1.05, 0.05)
        ax1.set_yticks(ticks)
        ax1.set_xticks(ticks)
        ax1.set_ylim((0, 1))
        ax1.set_xlim((0, 1))
        ax1.plot(ticks, ticks)
        ax1.plot(nn_fpr_keras, nn_thresholds_keras, label="Thresholds")
        # ax1.set_xticklabels(range(0, len(epochs)), fontsize=16)
        # ax1.set_yticklabels(ticks, fontsize=16)
        ax1.legend(fontsize=18)
        ax1.grid()
        plt.show()

    def evaluate(
        self,
        model,
        x_train=[],
        y_train=[],
        x_test=[],
        y_test=[],
        x_val=[],
        y_val=[],
        beta=1,
        threshold=0.5,
        print_metrics=True,
    ):
        train = Metrics()
        test = Metrics()
        val = Metrics()

        if x_train != []:
            y_predicted = self.__predict(model, x_train, y_train, threshold)
            tn, fp, fn, tp = self.__conf_matrix(y_train, y_predicted)
            train = self.__calc_metrics(tn, fp, fn, tp, beta)

        if x_test != []:
            y_predicted = self.__predict(model, x_test, y_test, threshold)
            tn, fp, fn, tp = self.__conf_matrix(y_test, y_predicted)
            test = self.__calc_metrics(tn, fp, fn, tp, beta)

        if x_val != []:
            y_predicted = self.__predict(model, x_val, y_val, threshold)
            tn, fp, fn, tp = self.__conf_matrix(y_val, y_predicted)
            val = self.__calc_metrics(tn, fp, fn, tp, beta)

        if print_metrics:
            text = textwrap.dedent(
                f"""\
                Exactitud: {train.accuracy} / {test.accuracy} / {val.accuracy}\n\
                Precisión: {train.precision} / {test.precision} / {val.precision}\n\
                Especificidad: {train.specifity} / {test.specifity} / {val.specifity}\n\
                Sensibilidad: {train.recall} / {test.recall} / {val.recall}\n\
                true negative rate: {train.true_negative_rate} / {test.true_negative_rate} / {val.true_negative_rate}\n\
                F - Score: {train.f_score} / {test.f_score} / {val.f_score}"""
            )
            print(text)

        return train, test, val


class Plotter:
    feature_names = (
        ["mfcc"] * 16
        + ["mfcc_delta1"] * 16
        + ["mfcc_delta2"] * 16
        + [
            "meanF0",
            "stdevF0",
            "meanF0delta",
            "hnr",
            "crest_factor",
            "rms",
            # "f_means",
            # "f_medians",
            "PCA",
        ]
    )
    plt.rcdefaults()

    def plot_feature_importance(self, n_top: int, feature_importance, std):

        _, ax = plt.subplots(figsize=(20, 20))
        top = np.argpartition(feature_importance, -n_top)[-n_top:]

        colors = np.array(["blue"] * len(feature_importance))
        colors[top] = "red"

        y_pos = np.arange(len(self.feature_names))
        ax.barh(y_pos, feature_importance, yerr=std, align="center", color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.feature_names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel("Feature Importance")
        ax.grid()
        plt.show()

    def correlation_plot(self, x, title):
        rho, pval = stats.spearmanr(x)

        fig, ax = plt.subplots(figsize=(20, 20))
        _ = ax.imshow(rho)
        n_features = len(self.feature_names)
        # We want to show all ticks...
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        # ... and label them with the respective list entries
        ax.set_xticklabels(self.feature_names)
        ax.set_yticklabels(self.feature_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(n_features):
            for j in range(n_features):
                text = ax.text(
                    j, i, round(rho[i, j], 2), ha="center", va="center", color="w"
                )

        ax.set_title("Correlación de Spearman del dataset  {}")
        fig.tight_layout()

        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(pval)

        # We want to show all ticks...
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        # ... and label them with the respective list entries
        ax.set_xticklabels(self.feature_names)
        ax.set_yticklabels(self.feature_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(n_features):
            for j in range(n_features):
                text = ax.text(
                    j, i, round(pval[i, j], 2), ha="center", va="center", color="w"
                )

        ax.set_title("p-valor para las correlaciones calculadas")
        fig.tight_layout()
        plt.show()

    def plot_loss(self, train_loss, valid_loss):

        _, ax1 = plt.subplots(figsize=(20, 10))
        color = "tab:blue"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=color)
        ax1.plot(train_loss, "--", color=color, label="Train Loss")
        ax1.plot(valid_loss, color=color, label="Valid Loss")
        ax1.tick_params(axis="y", labelcolor=color)
        plt.legend(loc="upper left")
        plt.title("Model Loss")
        plt.grid()
        plt.show()

    def plot_model_recall_fpr(self, train_recall, valid_recall):

        fig, ax1 = plt.subplots(figsize=(20, 10))
        color = "tab:red"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Recall", color=color)
        ax1.set_ylim([-0.05, 1.05])
        ax1.plot(train_recall, "--", color=color, label="Train Recall")
        ax1.plot(valid_recall, color=color, label="Valid Recall")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_yticks(np.linspace(0, 1, 40))
        plt.legend(loc="upper left")
        plt.title("Model Recall and FPR")

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()

    @classmethod
    def change_feature_names(cls, new_feature_names):
        cls.feature_names = new_feature_names