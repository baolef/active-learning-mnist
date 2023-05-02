from typing import Callable
from data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from modAL.models.base import BaseEstimator
from main import uncertainty_sampling
from modAL.models import ActiveLearner
from tqdm import tqdm


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot(model, dataset, filename, i):
    X = dataset.model.fit_transform(dataset.test_x)
    y = dataset.test_y
    # title for the plots
    title = 'Decision surface of linear SVC '
    # Set-up grid for plotting.

    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    h = (x_max - x_min) / 500
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    reshaped_grid = np.zeros((grid.shape[0], 784),  dtype=np.float64)

    reshaped_grid[:, 0] = grid[:, 0]
    reshaped_grid[:, 1] = grid[:, 1]

    Z = model.predict(reshaped_grid)
    Z = Z.reshape(xx.shape)
    Z = np.array(Z)
    Z = Z.astype(int)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], s=1, c=y.astype(int))

    plt.ylabel('y label here')
    plt.xlabel('x label here')
    plt.title('rbf decision boundary sample ' + str(i*10))
    plt.savefig(filename)
    plt.show()
    plt.close()


def learning(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, model: BaseEstimator,
             query: Callable, base: int = 10, samples: int = 40, batch: int = 3) -> list[float]:
    """
    Perform online learning with different query strategy and returns test accuracy.
    :param train_x: Train x.
    :param train_y: Train y.
    :param test_x: Test x.
    :param test_y: Test y.
    :param model: Base model in sklearn.
    :param query: Query strategy.
    :param base: The number of base training points.
    :param samples: The number of training points to be sampled.
    :param batch: Batch size.
    :return: Test accuracies of 30 rounds.
    """
    accuracy = []
    learner = ActiveLearner(
        estimator=model,
        query_strategy=query
    )
    learner.fit(train_x[:base], train_y[:base])
    train_x = train_x[base:]
    train_y = train_y[base:]
    accuracy.append(learner.score(test_x, test_y))

    for j, _ in enumerate(tqdm(range(samples // batch))):
        i, x = learner.query(train_x, n=batch)
        learner.teach(x, train_y[i])
        train_x = np.delete(train_x, i, axis=0)
        train_y = np.delete(train_y, i, axis=0)
        accuracy.append(learner.score(test_x, test_y))
        if j >= 50 and j <= 65:
            plot(model, dataset, 'decision_boundary_' + str(j) + '.png', j)

    return accuracy


def pipeline2(dataset: Dataset, model: BaseEstimator, query: Callable, label: str, base: int = 100, samples: int = 900,
             batch: int = 1, n: int = 1) -> None:
    """
    Train model in active learning with query strategy given dataset Output the accuracy to filename.
    :param dataset: Dataset.
    :param model: Base model in sklearn.
    :param query: Query strategy.
    :param label: Method label.
    :param base: The number of base training points.
    :param samples: The number of training points to be sampled.
    :param batch: Batch size.
    :param n: The number of experiments of the model.
    :return: None
    """
    np.random.seed(2023)
    acc = []
    train_x, test_x, train_y, test_y = dataset.get()
    for i in range(n):
        print('{}: round {}'.format(label, i + 1))
        train_x, train_y = shuffle(train_x, train_y)
        accuracy = learning(train_x, train_y, test_x, test_y, model, query, base, samples, batch)
        acc.append(accuracy)

    acc = np.array(acc)
    np.save('save/{}-{}-{}-{}.npy'.format(label, base, samples, batch), acc)
    # plot(acc, label, base, batch)


if __name__ == '__main__':
    dataset = Dataset()

    pipeline2(dataset, SVC(probability=True), uncertainty_sampling, 'uncertainty', 100, 900, 10, 2)


