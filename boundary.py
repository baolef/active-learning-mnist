# Created by Baole Fang at 2/15/23
import os
from typing import Callable
from data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from modAL.models import ActiveLearner
from modAL.models.base import BaseEstimator
from tqdm import tqdm
from main import uncertainty_sampling, density_sampling, diversity_sampling, random_sampling, minimize_expected_risk


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
    dataset.boundary(model, os.path.join('boundary', config_name, method, str(base) + '.png'))
    for iter in tqdm(range(samples // batch)):
        i, x = learner.query(train_x, n=batch)
        learner.teach(x, train_y[i])
        train_x = np.delete(train_x, i, axis=0)
        train_y = np.delete(train_y, i, axis=0)
        accuracy.append(learner.score(test_x, test_y))
        dataset.boundary(model, os.path.join('boundary', config_name, method, str(base + (iter + 1) * batch) + '.png'))
    print(accuracy)
    return accuracy


def plot(accuracy: np.ndarray, label: str, base: int = 0, batch: int = 3) -> None:
    """
    Plot the errorbar graph of accuracy.
    :param accuracy: accuracy.
    :param label: Method label.
    :param base: The starting count of legends.
    :param batch: Batch size.
    :return: None
    """
    legends = list(range(base, base + accuracy.shape[1] * batch, batch))
    plt.errorbar(legends, accuracy.mean(0), accuracy.std(0), capsize=3, label=label)
    plt.xlabel('samples')
    plt.ylabel('accuracy')
    plt.legend()


def pipeline(dataset: Dataset, model: BaseEstimator, query: Callable, name: str, label: str, base: int = 100,
             samples: int = 900,
             batch: int = 1, n: int = 1) -> None:
    """
    Train model in active learning with query strategy given dataset Output the accuracy to filename.
    :param dataset: Dataset.
    :param model: Base model in sklearn.
    :param query: Query strategy.
    :param name: Model config name.
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
        train_x, train_y = shuffle(train_x, train_y, random_state=2023 + i)
        accuracy = learning(train_x, train_y, test_x, test_y, model, query, base, samples, batch)
        acc.append(accuracy)


if __name__ == '__main__':
    dataset = Dataset()

    # config = {'C': 100, 'kernel': 'poly', 'degree': 3, 'probability': True}
    config = {'probability': True}
    # config_name='opt'
    config_name = 'default'
    # method='density'
    method = 'uncertainty'

    if not os.path.exists(os.path.join('boundary', config_name, method)):
        os.makedirs(os.path.join('boundary', config_name, method))

    # pipeline(dataset, SVC(**config), density_sampling, config_name, method, 100, 900, 10, 1)
    pipeline(dataset, SVC(**config), uncertainty_sampling, config_name, method, 100, 900, 10, 1)
