from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
from math import atan2, pi
import plotnine as pn
import pandas as pd


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(f"../{filename}")
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X_orig, y_orig = load_dataset(f"datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def call_me(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit._loss(X_orig, y_orig))

        perci = Perceptron(callback=call_me)
        perci.fit(X_orig, y_orig)

        # Plot figure of loss as function of fitting iteration
        df = pd.DataFrame({"iteration": range(len(losses)), "losses": losses})
        g = pn.ggplot(df, pn.aes("iteration", "losses")) + pn.geom_point(color="gray") + pn.theme_classic()
        g = g + pn.ggtitle(n)
        # print(g)
        pn.ggsave(g, f"../output/losses_{n}.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 1000)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return mu[0] + xs, mu[1] + ys


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X_orig, y_orig = load_dataset(f"datasets/{f}")

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X_orig, y_orig)
        lda_pred = lda.predict(X_orig)

        gnb = GaussianNaiveBayes()
        gnb.fit(X_orig, y_orig)
        gnb_pred = gnb.predict(X_orig)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA
        # predictions on the right. Plot title should specify dataset used and subplot titles should specify
        # algorithm and accuracy
        # Create subplots
        # Add traces for data-points setting symbols and colors
        from IMLearn.metrics import accuracy
        acc_lda = accuracy(y_orig, lda_pred)
        acc_gnb = accuracy(y_orig, gnb_pred)

        type1 = f"LDA\naccuracy: {round(acc_lda, 3)}"
        df1 = pd.DataFrame(
            {"y_true": y_orig.astype(str), "y_pred": lda_pred.astype(str),
             "feature_1": X_orig[:, 0],
             "feature_2": X_orig[:, 1], "type": type1})

        type2 = f"gaussian_naive_bayes\naccuracy: {round(acc_gnb, 3)}"
        df2 = pd.DataFrame(
            {"y_true": y_orig.astype(str), "y_pred": gnb_pred.astype(str),
             "feature_1": X_orig[:, 0],
             "feature_2": X_orig[:, 1], "type": type2})

        df = pd.concat([df1, df2])

        g = pn.ggplot(df) + pn.geom_point(pn.aes("feature_1", "feature_2", color="y_pred",
                                                 shape="y_true")) + pn.theme_classic()
        g = g + pn.ggtitle(f"data set: {f}")
        df_annotate = pd.DataFrame(
            {"x": np.concatenate((lda.mu_[:, 0], gnb.mu_[:, 0])),
             "y": np.concatenate((lda.mu_[:, 1], gnb.mu_[:, 1])),
             "type": np.repeat(np.array([type1, type2]), [3, 3]),
             "label": "x"})

        # Add `X` dots specifying fitted Gaussians' means
        g = g + pn.facet_grid(".~type") + pn.geom_text(data=df_annotate,
                                                       mapping=pn.aes(x=df_annotate["x"], y=df_annotate["y"],
                                                                      label=df_annotate["label"]))

        # Add ellipses depicting the covariances of the fitted Gaussians
        ellipse_df = None
        for i in range(3):
            lda_ellipse = get_ellipse(lda.mu_[i, :], lda.cov_)
            gnb_ellipse = get_ellipse(gnb.mu_[i, :], np.diag(gnb.vars_[i]))
            ellipse_df_lda = pd.DataFrame({"x": lda_ellipse[0], "y": lda_ellipse[1], "type": type1})
            ellipse_df_gnb = pd.DataFrame({"x": gnb_ellipse[0], "y": gnb_ellipse[1], "type": type2})
            if ellipse_df is None:
                ellipse_df = pd.concat([ellipse_df_lda, ellipse_df_gnb])
            else:
                temp = pd.concat([ellipse_df_lda, ellipse_df_gnb])
                ellipse_df = pd.concat([ellipse_df, temp])

        g = g + pn.geom_point(ellipse_df, pn.aes(x="x", y="y"), size=0.01)
        # print(g)
        pn.ggsave(g, f"../output/scatter_plot_{f[:-4]}.png")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
