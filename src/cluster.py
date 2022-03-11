import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_X_y
import math

from src.utils import is_power_of_two, binary_decomposition, extend_binary_decomposition


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
        score = metric(true_labels, copy_estimated_labels, **metric_params)
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
    def __init__(self, n_classes=None, n_leaves=None, prediction_type="deterministic", random_state=None,
                 logistic_params=None, random_empty=True):
        # parames
        self.n_classes = n_classes
        self.n_leaves = n_leaves
        self.prediction_type = prediction_type
        self.random_state = random_state
        self.logistic_params = logistic_params
        self.random_empty = random_empty
        # fitted attributes
        self.is_fitted_ = False
        self.classifiers_ = None
        self.leaf_classes_ = None
        self.leaf_class_probs_ = None
        self.cluster_leaves_association_ = None
        self.coef_ = None
        self.intercept_ = None


    def fit(self, X, y, cluster_labels=None, leaves_assignment=None):
        """
        This function fits the hierarchy of logistic classifiers. The y (target values) are use to assign
        class values to the leaf of the tree, while the cluster labels are used to perform the classification.
        :param X: Dataset on which the classifiers a re trained of shape (n_sample, n_features)
        :param y: Target of the classification of the hierarchy of shape (n_sample,)
        :param cluster_labels labelling assigned by an external actor. These labels are used to fit the individual
        classifiers
        :param leaves_assignment: Tells which cluster present in cluster_labels is assigned to which leaf
        :return: the Hierarchical Logistic Regressor fitted on the data
        """
        n_features = X.shape[1]

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
        # empty values
        if self.logistic_params is None:
            self.logistic_params = {}
        for i in range(n_logistics):
            if self.random_state is None:
                self.classifiers_.append(LogisticRegression(**self.logistic_params))
            else:
                self.classifiers_.append(LogisticRegression(random_state=self.random_state + i, **self.logistic_params))

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
            if np.unique(cl_filtered).size > 1:
                classifier = classifier.fit(x_filtered, cl_filtered)
            else:
                # TODO this may have some negative effects on precision with n_leaves greater than 4
                classifier = FixedBinaryClassificator(value=np.unique(cl_filtered)[0], n_features=n_features)
                classifier = classifier.fit(x_filtered, cl_filtered)

            # save the classifiers in the class
            trained_classifiers.append(classifier)
        self.classifiers_ = trained_classifiers

        # classifier coefficients
        coefficients = []
        intercepts = []
        for classifier in self.classifiers_:
            coefficients.append(classifier.coef_.squeeze())
            intercepts.append(classifier.intercept_.squeeze())
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
            if np.sum(classes_frequency) == 0:
                if self.random_empty:
                    self.leaf_class_probs_[i] = 1 / self.n_classes
                else:
                    self.leaf_class_probs_[i] = -1
            else:
                self.leaf_class_probs_[i] = classes_frequency / np.sum(classes_frequency)

        # correct -1 leaf_class_probs with adjacent leaf if not random empty
        for i in range(self.n_leaves):
            if not self.random_empty:
                if -1 in self.leaf_class_probs_[i]:
                    if i % 2 == 0:
                        self.leaf_class_probs_[i] = self.leaf_class_probs_[i+1].copy()
                    else:
                        self.leaf_class_probs_[i] = self.leaf_class_probs_[i-1]
            self.leaf_classes_[i] = np.argmax(self.leaf_class_probs_[i])


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
        if self.prediction_type == "deterministic":
            predictions = self.deterministic_predict(X)
        elif self.prediction_type == "probabilistic":
            predictions = self.probabilistic_predict(X)
        else:
            raise ValueError("The value {} of parameter prediction_type is not admissible".format(self.prediction_type))
        return predictions

    def deterministic_predict(self, X):
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

    def probabilistic_predict(self, X):
        class_probabilities = self.predict_proba(X)
        return np.argmax(class_probabilities, axis=1)

    def predict_proba(self, X):
        leaves_probabilities = self.leaves_probabilities(X)
        # class transformation can be done using self.leaf_class_probs that tells how to map leaf and classes
        return leaves_probabilities.dot(self.leaf_class_probs_)

    def score(self, X, y, sample_weight=None):
        return super().score(X, y, sample_weight)

    def leaves_probabilities(self, X):
        n_digits = int(math.log(self.n_leaves, 2))  # number of digits that can decode the tree
        leaves_probabilities = []
        for leaf_number in range(self.n_leaves):
            binary_list = binary_decomposition(leaf_number)
            binary_list = extend_binary_decomposition(binary_list, n_digits)
            leaf_prob = np.ones((X.shape[0],))
            classifier_index = 0
            digit_index = 0
            while digit_index < len(binary_list):
                digit = binary_list[digit_index]
                leaf_prob = leaf_prob * self.classifiers_[classifier_index].predict_proba(X)[:, digit]
                classifier_index = 2 * classifier_index + digit + 1
                digit_index += 1
            leaves_probabilities.append(leaf_prob)
        # leaves_probabilities now contains n_leaves arrays, and each array contains the probability of each sample
        # to fall in the leaf i
        # If we transpose this list, we get an array of shape (n_samples, n_leaves), and therefore each
        # row is the probability distribution of a given sample to fall in every leaf
        return np.array(leaves_probabilities).transpose()

    def get_ORCT_params(self, n_features, scale=512):
        a = np.stack(self.coef_).transpose()*n_features / 512
        mu = -np.stack(self.intercept_) / 512
        C = self.leaf_class_probs_.transpose()
        return {"a": a, "mu": mu, "C": C}


class FixedBinaryClassificator:

    def __init__(self, value, n_features):
        if value != 0 and value != 1:
            raise ValueError("Value is not 0 or 1. Insted, value={}".format(value))
        self.value = value
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        min_max_f = {
            0: np.min,
            1: np.max
        }
        f = min_max_f[self.value]
        n_features = X.shape[1]
        last_features = n_features - 1
        self.coef_ = np.zeros(n_features)
        self.coef_[-1] = 1
        self.intercept_ = np.ones(1) * f(X[-1])
        self.coef_ = np.random.uniform(0, 1, (n_features,))
        self.intercept_ = np.random.uniform(0, 1, (1,))
        return self

    def predict(self, X):
        return np.ones((X.shape[0],), dtype=int) * self.value

    def predict_proba(self, X):
        to_ret = np.zeros((X.shape[0], 2), dtype=int)
        to_ret[:, self.value] = 1
        return to_ret
