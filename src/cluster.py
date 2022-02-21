import numpy as np
import sklearn.base
from sklearn.metrics import completeness_score
import pandas as pd
import os
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
import numpy as np
from sklearn.metrics import rand_score, adjusted_rand_score, homogeneity_score, completeness_score
from sklearn.linear_model import LogisticRegression
import math
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y

from src.utils import is_power_of_two


def label_dataset(dataset, n_clusters, cluster_alg, alg_parameters, sample_weight):
    """
    This function fits the dataset for a generic cluster algorithm. When possible, it tries to
    have n_clusters in the output. However, algorithms with a smaller number of clusters are still allowed
    TODO add the possibility to split clusters in order to reach exactly N?
    TODO handle case in which clustering gives more than n_clusters clusters
    :param dataset: dataset that is fitted on the cluter algorithm
    :param n_clusters: number of clusters desired
    :param sample_weight: When possible, is given to the fit method of the cluster algorithm
    :param cluster_alg: algorithm
    :param alg_parameters: parameters of the cluster alghorithm
    :return:
    """
    assert len(dataset) == len(sample_weight)
    estimator = cluster_alg(**alg_parameters)
    try:
        estimator = estimator.fit(dataset, sample_weight=sample_weight)
    except TypeError:
        print(
            "The given cluster algorithm {} does not support sample_weight. Ignoring the argument".format(cluster_alg))
        estimator = estimator.fit(dataset)
    labels = estimator.labels_
    if len(np.unique(labels)) < n_clusters:
        print("The number of cluster found is less then to the number of cluster {} desired".format(n_clusters))
    elif len(np.unique(labels)) > n_clusters:
        NotImplementedError("Still not implemented case in which the number of clusters exceeds the desired one")
    return estimator


def find_best_estimator(estimator_list, metric, true_labels, maximize=True):
    scores = []
    for estimator in estimator_list:
        scores.append(metric(true_labels, estimator.labels_))
    if maximize:
        return estimator_list[np.argmax(scores)]
    else:
        return estimator_list[np.argmin(scores)]


def best_leaf_assignment(n_leaves, estimated_labels, true_labels, metric, metric_params=None):
    assert metric != None, "The metric is None"
    to_return = []  # ordering of the leaves from left to right
    clusters = list(range(n_leaves))
    n_active_clusters = len(clusters)
    estimated_labels = np.array(estimated_labels)  # deep copy
    n_runs = 0
    while n_active_clusters > 2:
        assignment, score = best_coupling([],
                                          clusters,
                                          estimated_labels=estimated_labels,
                                          true_labels=true_labels,
                                          metric=metric,
                                          metric_params=metric_params)
        # if it's the first run, the return value is the assignment
        if n_runs == 0:
            to_return = assignment
        # otherwise, the to_return computation is harder, since the assignment is referring to old assignments
        else:
            new_to_return = list()
            for couple in assignment:
                new_to_return.append(to_return[couple[0]] + to_return[couple[1]])
            to_return = new_to_return
        # now assign reassign labels, deepcopy to avoid errors while assigning values
        copy_estimated_labels = np.copy(estimated_labels)
        for i in range(len(to_return)):
            for cluster_index in to_return[i]:
                copy_estimated_labels[estimated_labels == cluster_index] = i
        estimated_labels = copy_estimated_labels
        n_active_clusters = len(to_return)
        clusters = list(range(n_active_clusters))
        n_runs += 1

    # flatting the return list
    to_return = to_return[0] + to_return[1]
    return to_return, score


def best_coupling(couples, remaining_clusters, estimated_labels, true_labels, metric, metric_params=None):
    # no more clusters to couple
    if metric_params is None:
        metric_params = {}
    if len(remaining_clusters) == 0:
        copy_estimated_labels = np.copy(estimated_labels)
        curr_couple = 0
        for couple in couples:
            copy_estimated_labels[estimated_labels == couple[0]] = curr_couple
            copy_estimated_labels[estimated_labels == couple[1]] = curr_couple
            curr_couple += 1
        score = metric(copy_estimated_labels, true_labels, **metric_params)
        return couples, score
    # still need to assign a cluster
    best_assignment = []
    best_score = 0
    for i in range(1, len(remaining_clusters)):
        new_couple = [remaining_clusters[0], remaining_clusters[i]]
        new_remaining_clusters = list(remaining_clusters)
        new_remaining_clusters.remove(remaining_clusters[0])
        new_remaining_clusters.remove(remaining_clusters[i])
        new_couples = list(couples)
        new_couples.append(new_couple)
        res_couples, score = best_coupling(new_couples,
                                           new_remaining_clusters,
                                           estimated_labels=estimated_labels,
                                           true_labels=true_labels, metric=metric)
        if score >= best_score:
            best_assignment = res_couples
            best_score = score
    return best_assignment, best_score


class HierarchicalLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=None, n_leaves=None, random_state=None):
        self.n_classes = n_classes
        self.n_leaves = n_leaves
        self.random_state = random_state
        self.is_fitted_ = False
        self.classifiers_ = None
        self.leaf_classes_ = None
        self.leaf_class_probs_ = None
        self.cluster_leaves_association_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y=None, cluster_labels=None, leaves_assignment=None):
        """
        This function fits the hierarchy of logistic classifiers. The y (target values) are use to assign
        class values to the leaf of the tree, while the cluster labels are used to perform the classification.
        :param x: Dataset on which the classifiers a re trained of shape (n_sample, n_features)
        :param y: Target of the classification of the hierarchy of shape (n_sample,)
        :param cluster_labels labelling assigned by an external actor. These labels are used to fit the individual
        classifiers
        :param leaves_assignment: Tells which cluster present in cluster_labels is assigned to which leaf
        :return: the Hierarchical Logistic Regressor fitted on the data
        """

        # checking X and y sizes
        X, y = check_X_y(X, y)

        # checking cluster labels and X
        try:
            X, cluster_labels = check_X_y(X, cluster_labels)
        except ValueError as v_error:
            raise ValueError("Problem between X and cluster_labels.\n Original exception: {}".format(v_error))

        # creation of the logistic regressors
        n_logistics = self.n_leaves - 1
        self.classifiers_ = list()
        for i in range(n_logistics):
            if self.random_state is None:
                self.classifiers_.append(LogisticRegression())
            else:
                self.classifiers_.append(LogisticRegression(random_state=self.random_state + i))

        # initialization of the value of the classes at the leaves and their probabilities
        if self.n_classes is None:
            raise ValueError("n_classes is None")
        if self.n_classes < 2:
            raise ValueError("n_classes value is {}, which is less then 2".format(self.n_classes))
        if self.n_leaves is None:
            raise ValueError("n_leaves is None")
        if not is_power_of_two(self.n_leaves):
            raise ValueError("n_leaves is not a power of 2")
        self.leaf_classes_ = -np.ones(self.n_leaves)
        self.leaf_class_probs_ = np.zeros((self.n_leaves, self.n_classes))

        self.assign_classes_to_leaves(y, cluster_labels, leaves_assignment)
        self.set_regressor_clusters(leaves_assignment)

        # hierarchical training of the clusters
        trained_classifiers = []
        for classifier, clust_couple in zip(self.classifiers_, self.cluster_leaves_association_):
            # filter the samples that are considered by the current clusters
            x_cluster_samples = []
            cl_cluster_samples = []
            for cluster_number in clust_couple[0] + clust_couple[1]:
                # this filtering should have no need of deep copying
                x_cluster_samples.append(X[y == cluster_number])
                cl_cluster_samples.append(y[y == cluster_number])
            x_filtered = np.concatenate(x_cluster_samples)
            cl_filtered = np.concatenate(cl_cluster_samples)
            copy_y_filtered = np.copy(cl_filtered)  # deep copy in order to assign target values
            # assign target values
            for i in range(len(clust_couple)):
                for cluster_value in clust_couple[i]:
                    cl_filtered[copy_y_filtered == cluster_value] = i
            classifier = classifier.fit(x_filtered, cl_filtered)
            # save the classifiers in the class
            trained_classifiers.append(classifier)
        self.classifiers_ = trained_classifiers

        # classifier coefficients
        coefficients = []
        intercepts = []
        for classifier in self.classifiers_:
            coefficients.append(classifier.coef_)
            intercepts.append(classifier.intercept_)
        self.coef_ = coefficients
        self.intercept_ = intercepts
        self.is_fitted_ = True

        return self

    def assign_classes_to_leaves(self, y, cluster_assignment, leaves_assignment):
        for i in range(self.n_leaves):
            cluster_leaf = leaves_assignment[i]
            y_cluster = y[cluster_assignment == cluster_leaf]
            classes_frequency = []
            for class_value in range(self.n_classes):
                classes_frequency.append(len(y_cluster[y_cluster == class_value]))
            self.leaf_classes_[i] = np.argmax(classes_frequency)
            self.leaf_class_probs_[i] = np.array(classes_frequency) / np.sum(classes_frequency)

    def set_regressor_clusters(self, leaves_assignment):
        cluster_dimension = self.n_leaves // 2
        n_of_adds = 1
        cluster_association = []
        while cluster_dimension > 0:
            i = 0
            while i < 2 * n_of_adds:
                start = i * cluster_dimension
                mid = i * cluster_dimension + cluster_dimension
                end = i * cluster_dimension + 2 * cluster_dimension
                cluster_association.append([leaves_assignment[start:mid], leaves_assignment[mid:end]])
                i += 2
            n_of_adds *= 2
            cluster_dimension = cluster_dimension // 2
        self.cluster_leaves_association_ = cluster_association

    def get_params(self, deep=True):
        ret_dict = dict(n_classes=self.n_classes,
                        n_leaves=self.n_leaves,
                        random_state=self.random_state)
        if deep:
            if self.classifiers_ is not None:
                for i in range(len(self.classifiers_)):
                    for key, value in self.classifiers_[i].get_params().items():
                        ret_dict["classifier_{}_{}".format(i, key)] = value
        return dict

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        predictions = list()
        for sample in X:
            # reshape needed since it's one sample ad a time but predict needs 2D arrays
            sample = sample.reshape(1, -1)
            classifier_index = 0
            while classifier_index < len(self.classifiers_):
                prediction = self.classifiers_[classifier_index].predict(sample)[0]
                # the classifiers_ is an array that mimics the structure of a binary tree.
                # The left child of node i is 2*i + 1, while the right child is 2*i+2.
                classifier_index = 2 * classifier_index + prediction + 1
            # still using binary trees property
            leaf = classifier_index - len(self.classifiers_)
            predictions.append(self.leaf_classes_[leaf])
        return np.array(predictions)

    def score(self, X, y, sample_weight=None):
        return super().score(X, y, sample_weight)
