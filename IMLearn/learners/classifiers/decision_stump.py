from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is above the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_err = np.inf
        for i in range(X.shape[1]):

            thr_pos, thr_err_pos = self._find_threshold(X[:, i], y, 1)
            thr_neg, thr_err_neg = self._find_threshold(X[:, i], y, -1)

            if thr_err_pos < min_err:
                min_err = thr_err_pos
                self.threshold_ = thr_pos
                self.sign_ = 1
                self.j_ = i

            if thr_err_neg < min_err:
                min_err = thr_err_neg
                self.threshold_ = thr_neg
                self.sign_ = -1
                self.j_ = i

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

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        true_vec = X[:, self.j_] >= self.threshold_
        res = np.full(X.shape[0], -1)
        res[true_vec] = 1
        return res * self.sign_

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        size = labels.size
        new_idx = np.argsort(values)
        sorted_values = values[new_idx]
        sorted_labels = labels[new_idx]
        # create matrix of all possabilities
        mat = np.full((size, size + 1), -sign)
        iu = np.tril_indices(size)
        mat[iu] = sign

        res = sorted_labels @ mat
        max_idx = np.argmax(res)

        mis = float(np.sum((np.sign(sorted_labels) != np.sign(mat[:, max_idx])) * np.abs(sorted_labels)))

        if max_idx == 0:
            return -np.inf, mis
        if max_idx == size:
            return np.inf, mis
        return sorted_values[max_idx], mis

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
        return float(np.sum(self.predict(X) != y)) / y.size
