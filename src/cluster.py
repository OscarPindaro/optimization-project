import numpy as np
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
        # print(estimated_labels[0:10])
        score = metric(copy_estimated_labels, true_labels, **metric_params)
        # print(couples, score)
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
        # print(res_couples, score)
        if score >= best_score:
            best_assignment = res_couples
            best_score = score
    return best_assignment, best_score


class HierarchicalLogisticRegression:
    def __init__(self, n_classes, n_leaves, random_state):
        # in a binary tree with n_leaves, the number of branching nodes is n_leaves - 1
        self.n_leaves = n_leaves
        n_logistics = n_leaves - 1
        self.classifiers = []
        for i in range(n_logistics):
            self.classifiers.append(LogisticRegression(random_state=random_state+i))
        # value of the class ad the bottom
        self.n_classes = n_classes
        self.leaf_classes = -np.ones(n_leaves)
        self.leaf_class_probs = np.zeros((n_leaves, n_classes))


    def fit(self, x, y, cluster_labels, leaves_assignment):
        """
        This function fits the hierarchy of logistic classifiers. The y (target values) are use to assign
        class values to the leaf of the tree, while the cluster labels are used to perform the classification.
        :param x: Dataset on which the classifiers a re trained of shape (n_sample, n_features)
        :param y: Target of the classification of the hierarchy of shape (n_sample,)
        :param cluster_labels labelling assigned by an external actor. These labels are used to fit the individual
        classifiers
        :param leaves_assignment: Tells which cluster present in cluster_labels is assigned to which leaf
        :return:
        """
        logistic_clusters = self.get_regressor_clusters(leaves_assignment)
        trained_classifiers = []
        for classifier, clust_couple in zip(self.classifiers, logistic_clusters):
            # filter the samples that are considered by the current clusters
            x_cluster_samples = []
            cl_cluster_samples = []
            for cluster_number in clust_couple[0] + clust_couple[1]:
                # this filtering should have no need of deep copying
                x_cluster_samples.append(x[y == cluster_number])
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
        self.classifiers = trained_classifiers

    def assign_classes_to_leaves(self, y,):
        raise NotImplementedError()


    def get_regressor_clusters(self, leaves_assignment):
        cluster_dimension = self.n_leaves // 2
        n_of_adds = 1
        cluster_association = []
        while cluster_dimension > 0:
            i = 0
            while i < 2 * n_of_adds:
                start = i * cluster_dimension
                mid = i * cluster_dimension + cluster_dimension
                end = i * cluster_dimension + 2 * cluster_dimension
                print(start, mid, end)
                cluster_association.append([leaves_assignment[start:mid], leaves_assignment[mid:end]])
                i += 2
            n_of_adds *= 2
            cluster_dimension = cluster_dimension // 2
        return cluster_association


if __name__ == "__main__":
    DATASET_PATH = os.path.join("../datasets", "car.csv")
    names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "Classes"]
    df = pd.read_csv(DATASET_PATH, delimiter=";", header=0, names=names)
    df = df.convert_dtypes()
    TARGET_INDEX = df.shape[1] - 1
    # dictionary converting ordinal categories to values
    cost_dict = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
    doors_dict = {"2": 2, "3": 3, "4": 4, "5more": 5}
    persons_dict = {"2": 2, "4": 4, "more": 5}
    dimension_dict = {"small": 0, "med": 1, "big": 2}
    # buying
    df["buying"] = df["buying"].apply(lambda x: cost_dict[x])
    df["maint"] = df["maint"].apply(lambda x: cost_dict[x])
    df["doors"] = df["doors"].apply(lambda x: doors_dict[x])
    df["persons"] = df["persons"].apply(lambda x: persons_dict[x])
    df["lug_boot"] = df["lug_boot"].apply(lambda x: dimension_dict[x])
    df["safety"] = df["safety"].apply(lambda x: cost_dict[x])
    classes_encoder = preprocessing.LabelEncoder().fit(df["Classes"])
    df["Classes"] = classes_encoder.transform(df["Classes"])
    clustering_estimators = []
    SEED = 1234
    X = df[list(df.columns)[:-1]]
    y = df["Classes"]
    params = dict(n_clusters=4, random_state=SEED)
    kmeans = KMeans(**params)
    kmeans = kmeans.fit(X, sample_weight=np.zeros(len(df)))
    clustering_estimators.append(kmeans)
    params = dict(n_clusters=4, random_state=SEED, assign_labels="discretize", gamma=1)
    spectral = SpectralClustering(**params)
    spectral = spectral.fit(X)
    clustering_estimators.append(spectral)
    params = dict(n_clusters=4, linkage="single")
    agglomerate = AgglomerativeClustering(**params)
    agglomerate = agglomerate.fit(X)
    clustering_estimators.append(agglomerate)
    params = dict(n_clusters=4)
    birch = Birch(**params)
    birch = birch.fit(X)
    clustering_estimators.append(birch)

    true_values = df["Classes"]
    f = completeness_score
    for estimator in clustering_estimators:
        print(estimator.__class__.__name__, f(true_values, estimator.labels_))

    estimator = find_best_estimator(clustering_estimators, completeness_score, y)
    print("The best estimate is {}".format(estimator))
    print("best estimator", best_leaf_assignment(4, estimator.labels_, true_values, completeness_score))
    print("true value", best_leaf_assignment(4, true_values, true_values, completeness_score))
    print(estimator.labels_)
    n_leaves = 4
    HLR = HierarchicalLogisticRegression(n_leaves, 0)
    ass, score = best_leaf_assignment(n_leaves, estimator.labels_, true_values, completeness_score)
    HLR.fit(X, y, ass)
