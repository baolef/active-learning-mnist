# Created by Baole Fang at 2/15/23

from typing import Callable
from data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from modAL.models import ActiveLearner
from modAL.models.base import BaseEstimator
import os
import multiprocess as mp
from copy import deepcopy
import scipy as sp
import timeit

def minimize_expected_risk(classifier: ActiveLearner, X_pool: np.ndarray, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedily samples n samples that minimize the expected risk from X_pool and return their indexes and values.
    :param classifier: The classifier for which the labels are to be queried.
    :param X_pool: The sample pool.
    :param n: The number of queries.
    :return: n random samples indexes and n random samples.
    """
    proc_pool = mp.Pool()

    print("Initialised pool")

    base_model = classifier.estimator
    n_classes = len(np.unique(classifier.y_training))
    X_u_prob = base_model.predict_proba(X_pool)
    X_pool_cp = np.copy(X_pool)
    tmp_model = deepcopy(classifier.estimator)


    print(len(X_pool))

    start = timeit.default_timer()
    print(start)

    def inner_pool_helper(i:int) -> float:
        start = timeit.default_timer()
        loss = 0
        for label in range(n_classes):
            tmp_x = np.append(classifier.X_training, [X_pool_cp[i,:]], axis = 0)
            tmp_y = np.append(classifier.y_training, [str(label)], axis = 0)
            tmp_model.fit(tmp_x, tmp_y)

            probs = tmp_model.predict_proba(X_pool_cp)
            prob_entropy = sp.stats.entropy(probs.T)
            loss += np.sum(prob_entropy) * X_u_prob[i,label]
        
        stop = timeit.default_timer()
        print('Time taken in seconds =',stop-start)
        
        return loss
    
    expected_risk = proc_pool.map(inner_pool_helper,range(len(X_pool)))
    proc_pool.close()
    proc_pool.join()

    

    if n == 1:
        idx = np.argmin(expected_risk)
        return idx, X_pool[idx]
    else:
        idx = np.argsort(expected_risk)[:n]
        return idx. X_pool[idx]




def uncertainty_sampling(classifier: ActiveLearner, X_pool: np.ndarray, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples n samples with the largest uncertainties from X_pool and return their indexes and values.
    :param classifier: The classifier for which the labels are to be queried.
    :param X_pool: The sample pool.
    :param n: The number of queries.
    :return: n random samples indexes and n random samples.
    """
    uncertainty = 1 - np.max(classifier.predict_proba(X_pool), axis=1)
    idx = np.argsort(uncertainty)[-n:]
    return idx, X_pool[idx]


def random_sampling(classifier: ActiveLearner, X_pool: np.ndarray, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly samples n samples from X_pool and return their indexes and values.
    :param classifier: The classifier for which the labels are to be queried.
    :param X_pool: The sample pool.
    :param n: The number of queries.
    :return: n random samples indexes and n random samples.
    """
    query_idx = np.random.choice(range(len(X_pool)), n, False)
    return query_idx, X_pool[query_idx]


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

    for _ in range(samples // batch):
        i, x = learner.query(train_x, n=batch)
        learner.teach(x, train_y[i])
        train_x = np.delete(train_x, i, axis=0)
        train_y = np.delete(train_y, i, axis=0)
        accuracy.append(learner.score(test_x, test_y))
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


def pipeline(dataset: Dataset, model: BaseEstimator, query: Callable, label: str, base: int = 100, samples: int = 900,
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
    for _ in range(n):
        train_x, train_y = shuffle(train_x[:,:2], train_y)
        accuracy = learning(train_x, train_y, test_x[:,:2], test_y, model, query, base, samples, batch)
        acc.append(accuracy)
    acc = np.array(acc)
    plot(acc, label, base, batch)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.plot('data.png')

    pipeline(dataset, SVC(probability=True), minimize_expected_risk, 'min_exp_risk', 10, 90, 1, 2)
    # pipeline(dataset, SVC(probability=True), random_sampling, 'passive', 10, 90, 1, 2)
    # pipeline(dataset, SVC(probability=True), uncertainty_sampling, 'uncertainty', 10, 90, 1, 2)

    plt.savefig('result.png')
    plt.close()
