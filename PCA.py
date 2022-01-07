import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin


class PCA(BaseEstimator, TransformerMixin):
# class PCA():
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        classes = np.unique(y)
        cov = []
        _ , channels, _ = X.shape

        for i in classes:
            x_class = X[y == i]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(channels, -1)
            x_class -= np.average(x_class, axis=1)[:, None]
            # get cov matrice for data in multiple channels
            # see https://www.py4u.net/discuss/190352
            cov_mat = np.squeeze(np.dot(x_class, x_class.T) / (x_class.shape[1] - 1)) 
            cov.append(cov_mat)
        covs = np.stack(cov)

        # Calculating Eigenvalues and Eigenvectors of the covariance matrix
        # eigen_values , eigen_vectors = np.linalg.eigh(covs[0], covs.sum(0))
        eigen_values , eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        # sort the eigenvalues/vector
        sorted_index = np.argsort(np.abs(eigen_values - 0.5))[::-1]
        sorted_eigenvectors = eigen_vectors[:,sorted_index].T
        # create list of new axes for projection
        self.new_axes = sorted_eigenvectors[:self.n_components]
        return self

    def transform(self, X):
        # Project X data on new axes 
        X_reduce = np.asarray([np.dot(self.new_axes, i) for i in X])
        # get new data
        X_reduce = (X_reduce ** 2).mean(axis=2)

        return X_reduce

    def fit_transform(self, X, y):
        self.fit(X,y)
        ret = self.transform(X)
        return ret


# PCA:
# https://www.youtube.com/watch?v=uV5hmpzmWsU 
# https://www.askpython.com/python/examples/principal-component-analysis

# PIPELINE OBJECT:
# https://ichi.pro/fr/transformateurs-personnalises-et-pipelines-de-donnees-ml-avec-python-36190107130725

# COVAR MATRICE:
# https://www.py4u.net/discuss/190352 