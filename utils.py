""" Utility functions for scikit learn models models """

import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=invalid-name, line-too-long

def plot_decision_boundary(X, Y, predict, title="Decision Boundary", n_cells=50):
    """ Plots a trained models decision boundary for a binary classification, alongside sample data

    :param X: A 2d nparray of features
    :param Y: A 1d nparray of classifications; either 0 or 1
    :param n_cells: Number of cells in the grid used to draw the decision boundary
    :param predict: Callable that can be used to make predictions
    """

    if len(X.shape) != 2 or X.shape[1] != 2:
        raise ValueError("X must be ann nx2 array of examples")
    if len(Y.shape) != 1:
        raise ValueError("Y must be an 1 dimensional array of classifications")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    if not np.all(np.logical_or(Y == 0, Y == 1)):
        raise ValueError("All values of Y must be either 0 or 1")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1     # The range of X values (first col) in X
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1     # The range of Y values (2nd col) in X

    # Create a 50x50 mesh grid over the range of X and Y values, and draw a decision boundary based on the
    # predicted classification for each cell
    xx = np.linspace(x_min, x_max, n_cells)                 # x values of the grid
    yy = np.linspace(y_min, y_max, n_cells)                 # y values of the grid
    y_hat = np.zeros((n_cells, n_cells))                    # predicted value of each cell in the grid
    for i in range(n_cells):
        for j in range(n_cells):
            y_hat[j, i] = predict([[xx[i], yy[j]]])
    plt.contourf(xx, yy, y_hat, alpha=0.8, cmap = "Blues")  # Plot the decision boundary over the mesh grid

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap = "summer", s=40)   # Plot the actual classifications

    plt.title(title)

    plt.show()
