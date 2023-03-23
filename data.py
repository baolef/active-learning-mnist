# Created by Baole Fang at 3/23/23

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Dataset:
    def __init__(self, train_ratio=0.8, scale=True):
        X, y = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
        )
        scaler = StandardScaler()
        if scale:
            X=scaler.fit_transform(X)
        train_size=int(train_ratio*len(X))

        self.n_features = X.shape[1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=train_size, test_size=len(X)-train_size
        )

    def get_train(self,n_features=0):
        if 0<n_features<self.n_features:
            pca=PCA(n_components=n_features)
            return pca.fit_transform(self.X_train), self.y_train
        return self.X_train, self.y_train

    def get_test(self,n_features=0):
        if 0<n_features<self.n_features:
            pca=PCA(n_components=n_features)
            return pca.fit_transform(self.X_test), self.y_test
        return self.X_test, self.y_test


if __name__ == '__main__':
    dataset=Dataset()
    X,y=dataset.get_train()
    X,y=dataset.get_test()
