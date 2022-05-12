from typing import Tuple

from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotnine as pn
from IMLearn.metrics.loss_functions import accuracy
import numpy as np
import pandas as pd


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def decision_surface(predict, xrange, yrange, t, X, density=250):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], t)
    return xx.ravel(), yy.ravel(), pred


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train - and test errors of AdaBoost in noiseless case
    ada = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners)
    ada.fit(train_X, train_y)
    train_loss = [ada.partial_loss(train_X, train_y, i) for i in range(1, n_learners)]
    test_loss = [ada.partial_loss(test_X, test_y, i) for i in range(1, n_learners)]
    num_of_iter = [i for i in range(1, n_learners)]
    df_1 = pd.DataFrame({"num_of_iter": num_of_iter, "loss": test_loss, "type": "test_loss"})
    df_2 = pd.DataFrame({"num_of_iter": num_of_iter, "loss": train_loss, "type": "train_loss"})
    df = pd.concat([df_1, df_2])
    g = pn.ggplot(df) + pn.geom_line(pn.aes("num_of_iter", "loss", color="type")) + pn.theme_classic()
    # print(g)
    pn.ggsave(g, f"../output/scatter_plot_{noise}.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    df_1_all = None
    df_2_all = None
    for t in T:
        x, y, z = decision_surface(ada.partial_predict, lims[0], lims[1], t, test_X)
        df_1 = pd.DataFrame({"x": x, "y": y, "z": z.astype(str), "type": t})
        df_2 = pd.DataFrame(
            {"x": test_X[:, 0], "y": test_X[:, 1], "res": ada.partial_predict(test_X, t).astype(str),
             "type": t})
        df_1_all = pd.concat([df_1_all, df_1])
        df_2_all = pd.concat([df_2_all, df_2])

    g = pn.ggplot(None) + pn.geom_point(df_1_all,
                                        pn.aes("x", "y", color="z")) + pn.theme_classic()
    g = g + pn.geom_point(df_2_all, pn.aes("x", "y", shape="res"))
    g = g + pn.facet_wrap("~type", nrow=2)
    g = g + pn.ggtitle("Decision boundaries with test data") + pn.xlab("feature 0") + pn.ylab("feature 1")

    # print(g)
    pn.ggsave(g, f"../output/grid_plot_{noise}.png")

    # Question 3: Decision surface of best performing ensemble
    min_err_idx = int(np.argmin(np.array([test_loss])))
    x, y, z = decision_surface(ada.partial_predict, lims[0], lims[1], min_err_idx, test_X)
    df_1 = pd.DataFrame({"x": x, "y": y, "z": z.astype(str), "type": min_err_idx})
    y_pred = ada.partial_predict(test_X, min_err_idx)
    df_2 = pd.DataFrame(
        {"x": test_X[:, 0], "y": test_X[:, 1], "res": y_pred.astype(str),
         "type": min_err_idx})
    g = pn.ggplot(None) + pn.geom_point(df_1,
                                        pn.aes("x", "y", color="z")) + pn.theme_classic()
    g = g + pn.geom_point(df_2, pn.aes("x", "y", shape="res"))
    g = g + pn.ggtitle(
        f"Decision surface of best performing ensemble\naccuracy: {accuracy(test_y, y_pred)}\niteration: {min_err_idx}") + pn.xlab(
        "feature 0") + pn.ylab(
        "feature 1")

    # print(g)
    pn.ggsave(g, f"../output/min_plot_{noise}.png")

    # Question 4: Decision surface with weighted samples
    x, y, z = decision_surface(ada.partial_predict, lims[0], lims[1], n_learners, test_X)
    df_1 = pd.DataFrame({"x": x, "y": y, "z": z.astype(str), "type": min_err_idx})
    df_2 = pd.DataFrame(
        {"x": train_X[:, 0], "y": train_X[:, 1], "res": ada.partial_predict(train_X, n_learners).astype(str),
         "type": min_err_idx, "weights": ada.D_})
    g = pn.ggplot(None) + pn.geom_point(df_1,
                                        pn.aes("x", "y", color="z")) + pn.theme_classic()
    scale_fac = 1
    if noise == 0:
        scale_fac = 5
    scale = df_2["weights"] / np.max(df_2["weights"]) * scale_fac
    g = g + pn.geom_point(df_2,
                          pn.aes("x", "y", shape="res", size=scale))
    g = g + pn.ggtitle("Decision surface with weighted samples") + pn.xlab("feature 0") + pn.ylab(
        "feature 1") + pn.scale_size_continuous(range=(np.min(scale), np.max(scale)))

    # print(g)
    pn.ggsave(g, f"../output/weights_plot_{noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise, n_learners=250)
