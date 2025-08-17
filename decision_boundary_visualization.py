import numpy as np
from utils import Visualizer
from sklearn import svm
from typing import Callable


def polar_kernel_matrix(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Matrix output version of polar_kernel. Sklearn expects
    custom kernel functions to be in this format for performance purposes.

    Parameters
    ----------
    X1: np.ndarray
        A numpy matrix of size n_samples x features
    X2: np.ndarray
        A numpy matrix of the same dimensions as X1

    Returns
    -------
    np.ndarray
        A n_sample x n_sample matrix where each element is K(Xi, Xj)
    """
    squared_norm1 = np.linalg.norm(X1, axis=1) ** 2
    squared_norm2 = np.linalg.norm(X2, axis=1) ** 2
    return np.inner(X1, X2) + np.outer(squared_norm1, squared_norm2)


def polar_kernel(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Returns K(X1, X2) where K is the kernel represenation of <phi(X1), phi(X2)>
    and phi is the polar embedding.

    Parameters
    ----------
    X1: np.ndarray
        A numpy row vector, representing a single sample.
    X2: np.ndarray
        A numpy row vector, representing a single sample.

    Returns
    -------
    float
        A scalar value representing K(X1, X2).
    """
    return np.dot(X1, X2) + np.linalg.norm(X1) ** 2 * np.linalg.norm(X2) ** 2


def polar_embedding(X: np.ndarray) -> np.ndarray:
    """
    Returns X in higher dimensional feature space, in otherwords, adds a feature to each sample in X.
    The added feature should be the right most column.

    Parameters
    ----------
    X: np.ndarray
        A n_sample x n_features numpy matrix of samples. In part 1&2, X has 2 original features.

    Returns
    -------
    np.ndarray
        A n_sample x n_features + 1 matrix of samples with the additional feature appended on the right.
    """
    norm_squared = np.linalg.norm(X, axis=1) ** 2
    return np.hstack((X, norm_squared.reshape(-1, 1)))

class SolutionVisualization(Visualizer):
    def __init__(
        self, svm: svm, kernel: np.ndarray = None, embedding_function: Callable = None
    ):
        super().__init__()
        self.svm = svm
        self.kernel = kernel
        self.embedding_function = embedding_function

    def generate_new_feature_values(
        self, gridpoints_x1: np.ndarray, gridpoints_x2: np.ndarray
    ) -> np.ndarray:
        """
        Generates x_1^2 + x_2^2 values which lie on H(x) = 0 for each gridpoint (x_1, x_2).

        Parameters
        ----------
        gridpoints_x1: np.ndarray
            A n_axis_len x n_axis_len (see utils.py for defition) numpy matrix of values representing x_1 values.
        gridpoints_x2: np.ndarray
            A n_axis_len x n_axis_len numpy matrix of values representing x_2 values.

        Returns
        -------
        A n_axis_len x n_axis_len numpy matrix of values representing x_1^2 + x_2^2 values for each corresponding gridpoint
        """
        # the weight vector for H, shape: 1 x n_features
        coefficients = self.svm.coef_.flatten()
        # the scalar bias term
        intercept = self.svm.intercept_.flatten()

        # Calculate x2^2 + x1^2 points which lie on H(x)
        new_feature_points = (-coefficients[0] * gridpoints_x1 - coefficients[1] * gridpoints_x2 - intercept) / coefficients[2]
        return new_feature_points

    def _H(
        self,
        X: np.ndarray,
        x_input: np.ndarray,
        dual_coefficients: np.ndarray,
        intercept: float,
    ) -> np.ndarray:
        """
        Calculates H using formula (9) in the handout.

        Parameters
        ----------
        X: np.ndarray
            A n_support_vectors x n_features numpy array representing the training data samples which are support vectors.
        x_input: np.ndarray
            A 1 x n_features numpy array representing a single data point to evaluate H on.
        dual_coefficients: np.ndarray
            A numpy vector of coefficients of each X_data point, notated as alpha in the handout.
        intercept: float
            A scalar representing the bias term
        Returns
        -------
        np.ndarray
            A 1 x 1 numpy vector, which is the value of H(x_input_i).
        """
        # Compute K(X_i, x_input) for each support vector X_i
        kernel_vals = np.array([self.kernel(x_i, x_input) for x_i in X])
        # Implement the equation for H
        H = np.sum(dual_coefficients * kernel_vals) + intercept
        return H

    def generate_H_points(
        self, X: np.ndarray, gridpoints_x1: np.ndarray, gridpoints_x2: np.ndarray
    ) -> np.ndarray:
        """
        Generates an H value for each gridpoint (x_1, x_2).

        Parameters
        ----------
        X: np.ndarray
            A n_training_samples x n_features matrix of training data.
        gridpoints_x1: np.ndarray
            A n_axis_len x n_axis_len numpy matrix of values representing x_1 values.
        gridpoints_x2: np.ndarray
            A n_axis_len x n_axis_len numpy matrix of values representing x_2 values.

        Returns
        -------
        np.ndarray
            A n_axis_len x n_axis_len numpy matrix of values representing H values for each corresponding gridpoint.
        """
        dual_coefficients = np.zeros(len(X))
        support_indices = self.svm.support_
        dual_coefficients[support_indices] = self.svm.dual_coef_.flatten()
        # Get the value of H at each gridpoint
        # gridpoints here is a n_gridpoints x 2 matrix where each row is a (x_1, x_2) gridpoint
        gridpoints = np.c_[gridpoints_x1.ravel(), gridpoints_x2.ravel()]
        H = np.array([self._H(X[support_indices], point, dual_coefficients[support_indices], self.svm.intercept_) for point in gridpoints])

        return H.reshape(gridpoints_x1.shape)

    def visualize_H(
        self,
        X: np.ndarray,
        gridpoints_x1: np.ndarray,
        gridpoints_x2: np.ndarray,
        plot_params: dict = {},
    ) -> None:
        """
        Generates a plot of H(x).

        Parameters
        ----------
        X: np.ndarray
            A n_training_samples x n_features matrix of training data.
        gridpoints_x1: np.ndarray
            A n_axis_len x n_axis_len numpy matrix of values representing x_1 values.
        gridpoints_x2: np.ndarray
            A n_axis_len x n_axis_len numpy matrix of values representing x_2 values.
        plot_params: dict
            A dictionary containing parameters for the plot.

        Returns
        -------
        None (A plot should appear when called.)
        """
        H = self.generate_H_points(X, gridpoints_x1, gridpoints_x2)

        self.visualize_3d(gridpoints_x1, gridpoints_x2, H, plot_params=plot_params)
