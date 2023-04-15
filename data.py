# Created by Baole Fang at 3/23/23

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


class Dataset:
    def __init__(self, train_ratio: float = 0.8, scale: bool = True, n_features: int = 0, feature_extraction=PCA):
        '''
        Dataset initialization.
        :param train_ratio: The ratio of training set.
        :param scale: Whether to rescale the data.
        :param n_features: The number of features.
        :param feature_extraction: The feature extraction method, eg. PCA, TSNE.
        '''
        X, y = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
        )
        scaler = StandardScaler()
        if scale:
            X = scaler.fit_transform(X)

        if 0 < n_features < X.shape[1]:
            self.model = feature_extraction(n_features)
            X = self.model.fit_transform(X)
            self.visual_flag = False
        else:
            self.model = feature_extraction(2)
            self.visual_flag = True

        train_size = int(train_ratio * len(X))
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            X, y, train_size=train_size, test_size=len(X) - train_size, random_state=2023
        )

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Return all the data.
        :return: Data.
        '''
        return self.train_x, self.test_x, self.train_y, self.test_y

    def plot(self, filename, y=None) -> None:
        '''
        Plot the dataset.
        :param filename: The filename of the figure.
        :param y: The predicted y. If passed, then a true/false figure will be plotted indicating whether a test sample
        is predicted correctly. If not passed, then the original dataset will be plotted.
        :return: None.
        '''
        if y:
            y = y == self.test_y
        else:
            y = self.test_y
        x = self.test_x
        if self.visual_flag:
            x = self.model.fit_transform(x)
        plt.scatter(x[:, 0], x[:, 1], s=1, c=y.astype('int'))
        plt.savefig(filename)
        plt.close()
