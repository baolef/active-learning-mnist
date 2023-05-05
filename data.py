# Created by Baole Fang at 3/23/23

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from modAL.models import ActiveLearner


class Dataset:
    def __init__(self, train_size: int = 5000, test_size: int = 10000, scale: bool = True, n_features: int = 0,
                 feature_extraction=PCA):
        '''
        Dataset initialization.
        :param train_size: The size of the training set.
        :param test_size: The size of the test set.
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

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=2023
        )

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Return all the data.
        :return: Data.
        '''
        return self.train_x, self.test_x, self.train_y, self.test_y

    def plot(self, filename: str, y: np.ndarray = None) -> None:
        '''
        Plot the dataset.
        :param filename: The filename of the figure.
        :param y: The predicted y. If passed, then a true/false figure will be plotted indicating whether a test sample
        is predicted correctly. If not passed, then the original dataset will be plotted.
        :return: None.
        '''
        x = self.test_x
        if self.visual_flag:
            x = self.model.fit_transform(x)
        if y is not None:
            y = y == self.test_y
            plt.scatter(x[y, 0], x[y, 1], s=1, c=self.test_y[y].astype('int'), marker='.')
            plt.scatter(x[~y, 0], x[~y, 1], s=1, c=self.test_y[~y].astype('int'), marker='s')
        else:
            plt.scatter(x[:, 0], x[:, 1], s=1, c=self.test_y.astype('int'))
        plt.title(filename.rstrip('.png'))
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def boundary(self, model: ActiveLearner, filename: str) -> None:
        '''
        Plot the decision boundary of the model.
        :param model: The classification model.
        :param filename: Output filename.
        :return: None
        '''
        x = self.test_x
        if self.visual_flag:
            x = self.model.fit_transform(x)
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        h = (x_max - x_min) / 500
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid = self.model.inverse_transform(grid)
        z = model.predict(grid)
        z = np.array(z.reshape(xx.shape), dtype=int)
        plt.contourf(xx, yy, z)
        plt.scatter(x[:, 0], x[:, 1], s=1, c=self.test_y.astype(int))
        plt.savefig(filename)
        plt.close()

    def kmeans(self, filename: str, n: int = 10) -> None:
        '''
        Plot kmeans clustering result of the dataset.
        :param filename: The filename of the figure.
        :param n: The number of clusters.
        :return: None.
        '''
        x = self.test_x
        if self.visual_flag:
            x = self.model.fit_transform(x)
        if self.visual_flag:
            x = self.model.fit_transform(x)
        cluster = KMeans(n_clusters=n, random_state=0, n_init="auto")
        labels = cluster.fit_predict(self.test_x)
        u_labels = np.unique(labels)
        for i in u_labels:
            plt.scatter(x[labels == i, 0], x[labels == i, 1], label=i, s=1)
        plt.title(filename.rstrip('.png'))
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


if __name__ == '__main__':
    dataset = Dataset()
    dataset.plot('classes.png')
    dataset.kmeans('kmeans.png')
