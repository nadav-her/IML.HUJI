from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # X = np.array([1, 1, 1, 2, 2, 3, 2, 4, 3, 3, 3, 4]).reshape((6, 2)).astype(np.float64)
        # y = np.array([0, 0, 1, 1, 1, 1]).astype(np.int64)

        self.classes_ = np.unique(y)
        bins = np.bincount(y)[self.classes_]
        self.pi_ = bins / y.size
        sorted_idx = y.argsort()
        X_sorted = X[sorted_idx, :]
        ind = np.cumsum(np.insert(bins, 0, 0))[:-1]
        self.mu_ = np.add.reduceat(X_sorted, ind)
        self.mu_ /= bins[:, None]
        mu_expand = np.repeat(self.mu_, bins, axis=0)
        X_sorted_minus_mu = X_sorted - mu_expand
        self.vars_ = X_sorted_minus_mu[:, :, None] @ X_sorted_minus_mu[:, None, :]
        self.vars_ = np.add.reduceat(self.vars_, ind)
        bins -= 1
        self.vars_ /= bins[:, None, None]
        self.vars_ = np.diagonal(self.vars_, axis1=1, axis2=2)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pred = np.argmax(self.likelihood(X), axis=1)
        return self.classes_[pred]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        mat = np.zeros((X.shape[0], self.mu_.shape[0]))
        for i in range(self.vars_.shape[0]):
            var_k = np.diag(self.vars_[i])
            var_k_inv = np.linalg.inv(var_k)
            mu_k = self.mu_[i]
            pi_k = self.pi_[i]
            a = var_k_inv @ mu_k.T
            b = np.log(pi_k) - 0.5 * mu_k @ var_k_inv @ mu_k.T
            c = -0.5 * (np.diagonal(X @ var_k_inv @ X.T))
            z = -np.log((2 * np.pi) ** (X.shape[1] * 0.5) * np.linalg.det(var_k) ** 0.5)
            mat[:, i] = X @ a + b + c + z
        return mat

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        y_pred = self.predict(X)
        misclassification_error(y, y_pred)
