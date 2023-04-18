# Created by Baole Fang at 2/15/23

from typing import Callable
from data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from modAL.models import ActiveLearner
from modAL.models.base import BaseEstimator
from tqdm import tqdm


def uncertainty_sampling(classifier: ActiveLearner, X_pool: np.ndarray, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples n samples with the largest uncertainties from X_pool and return their indexes and values.
    :param classifier: The classifier for which the labels are to be queried.
    :param X_pool: The sample pool.
    :param n: The number of queries.
    :return: n indices from samples maximizing uncertainty and the n samples.
    """
    uncertainty = 1 - np.max(classifier.predict_proba(X_pool), axis=1)
    idx = np.argsort(uncertainty)[-n:]
    return idx, X_pool[idx]

def density_sampling(classifier: ActiveLearner, X_pool: np.ndarray, n: int = 1, beta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples n samples with the largest product of uncertainty-based utility and average similarity to other samples in X_pool, returning their indices and values.
    :param classifier: The classifier for which the labels are to be queried.
    :param X_pool: The sample pool.
    :param n: The number of queries.
    :param beta: Hyperparameter exponentiating the average similarity term of the metric.
    :return: n indices from samples maximizing the density-based metric and the n samples.
    """
    uncertainty = 1 - np.max(classifier.predict_proba(X_pool), axis=1)
    num_points = len(X_pool)
    avg_similarity = np.zeros(num_points)
    for i in range(num_points):
        dists = np.linalg.norm(np.delete(X_pool, i, 0) - X_pool[i], axis=1) # list of Euclidean distances between i-th sample and all others in pool
        avg_similarity[i] = np.mean(1.0 / (1.0 + dists)) # 1 / (1 + d) converts distance measure to similarity measure with maximum of 1
    idx = np.argsort(uncertainty * (avg_similarity ** beta))[-n:]
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

def diversity_sampling(classifier: ActiveLearner, X_pool: np.ndarray, n: int = 1):
    if X_pool.shape[0] > n:
        cluster = KMeans(n_clusters=n, random_state=0, n_init="auto")
        labels = cluster.fit_predict(X_pool)
        idx_pool = np.empty(n,dtype=int)

        for k in range(n):
            pool = np.argwhere(labels==k).reshape(-1)
            
            idx_pool[k] = np.random.choice(pool, size=1, replace=False)
        
    else:
        idx_pool = np.array(range(X_pool.shape[0]))

    return idx_pool, X_pool[idx_pool]


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

    for _ in tqdm(range(samples // batch)):
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
    for i in range(n):
        print('{}: round {}'.format(label,i+1))
        train_x, train_y = shuffle(train_x, train_y)
        accuracy = learning(train_x, train_y, test_x, test_y, model, query, base, samples, batch)
        acc.append(accuracy)
    acc = np.array(acc)
    plot(acc, label, base, batch)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.plot('data.png')

    pipeline(dataset, SVC(probability=True), diversity_sampling, 'diversity', 10, 90, 5, 2)
    pipeline(dataset, SVC(probability=True), random_sampling, 'passive', 10, 90, 1, 2)
    pipeline(dataset, SVC(probability=True), uncertainty_sampling, 'uncertainty', 10, 90, 1, 2)

    plt.savefig('result.png')
    plt.close()
