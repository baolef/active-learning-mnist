# Created by Baole Fang at 3/23/23

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Dataset:
    def __init__(self, train_ratio=0.8, scale=True):
        '''
        Dataset initialization.
        :param train_ratio: The ratio of training set.
        :param scale: Whether to rescale the data.
        '''
        X, y = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
        )
        scaler = StandardScaler()
        if scale:
            X = scaler.fit_transform(X)
        train_size = int(train_ratio * len(X))
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=train_size, test_size=len(X) - train_size
        )

    def pca(self, n_components):
        '''
        Fit PCA given n_components.
        :param n_components: The number of output features of PCA.
        :return: A fitted PCA that reduces data dimension to n_components.
        '''
        pca = PCA(n_components)
        pca.fit(self.X)
        return pca

    def get_train(self, n_features=0):
        '''
        Get training set with n_features features.
        :param n_features: The number of features.
        :return: Training set with n_features features.
        '''
        if 0 < n_features < self.n_features:
            pca = self.pca(n_features)
            return pca.transform(self.X_train), self.y_train
        return self.X_train, self.y_train

    def get_test(self, n_features=0):
        '''
        Get testing set with n_features features.
        :param n_features: The number of features.
        :return: Testing set with n_features features.
        '''
        if 0 < n_features < self.n_features:
            pca = self.pca(n_features)
            return pca.fit_transform(self.X_test), self.y_test
        return self.X_test, self.y_test


if __name__ == '__main__':
    dataset = Dataset()
    X, y = dataset.get_train()
    X, y = dataset.get_test()
    X, y = dataset.get_train(2)
    X, y = dataset.get_test(3)
