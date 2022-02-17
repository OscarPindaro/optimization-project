import numpy as np
from sklearn.metrics import completeness_score


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
    return estimator_list[np.argmax(scores)]


def assign_clusters_to_leaves(n_leaves, true_labels, estimated_labels):
    leaves = list(range(n_leaves))
    stack = []
    done = False
    while not done:
        copy_leaves = list(leaves)


def recursive_best_completeness(couples, remaining_clusters, estimated_labels, true_labels):
    # no more clusters to couple
    if len(remaining_clusters) == 0:
        estimated_labels = np.array(estimated_labels)
        curr_couple = 0
        for couple in couples:
            estimated_labels[estimated_labels == couple[0]] = curr_couple
            estimated_labels[estimated_labels == couple[1]] = curr_couple
            curr_couple += 1
        #print(estimated_labels[0:10])
        print(np.unique(estimated_labels))
        score = completeness_score(estimated_labels, true_labels)
        print(couples, score)
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
        res_couples, score = recursive_best_completeness(new_couples,
                                                         new_remaining_clusters,
                                                         estimated_labels=estimated_labels,
                                                         true_labels=true_labels)
        # print(res_couples, score)
        if score >= best_score:
            best_assignment = res_couples
            best_score = score
    return best_assignment, best_score


if __name__ == "__main__":
    pass
