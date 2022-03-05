#!/usr/bin/env python
# coding: utf-8

# # Thyroid Dataset
# ## Preprocessing

# In[1]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch
from sklearn.metrics import completeness_score, homogeneity_score, balanced_accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.ORCTModel import ORCTModel, predicted_lab, accuracy, pr
from src.cluster import HierarchicalLogisticRegression, best_leaf_assignment

import operator
from pyomo.environ import *
from pyomo.opt import SolverFactory

# In[2]:

if __name__ == "__main__":
    SEED = 1
    name = "seeds_data.csv"
    DATASET_PATH = os.path.join("datasets", name)
    df = pd.read_csv(DATASET_PATH, delimiter=";", header=0)
    le = preprocessing.LabelEncoder()
    le.fit(df['Classes'])

    df['Classes'] = le.transform(df['Classes'])

    columns = list(df.columns)
    X = df[columns[:-1]]
    y = df[columns[-1]]
    feature_names = columns[:-1]

    print("Number of rows: {}\nShape: {}".format(len(df), df.shape))
    print("The are {} columns".format(len(df.columns)))
    print("\nDistinct values for 'Classes' column\n{}\n".format(df["Classes"].value_counts()))

    vals = y.unique()
    vals.sort()
    heights = [len(y[y == x]) for x in vals]
    vals = [str(x) for x in vals]
    # plt.bar(vals, heights)
    # plt.show()
    X_std = X.copy()
    X_std[feature_names] = MaxAbsScaler().fit_transform(X[feature_names])
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.25, random_state=SEED)
    index_features = list(range(0, len(feature_names)))
    index_instances = list(X_train.index)
    df_train = pd.concat([X_train, y_train], axis=1)
    classes = y.unique().tolist()
    classes.sort()  # sorted
    classes_en = [i for i in range(len(classes))]

    occurences = [len(y_train[y_train == x]) for x in classes]
    total_samples = sum(occurences)
    sample_weight = np.zeros_like(y_train)
    for class_index, n_occurr in zip(classes, occurences):
        sample_weight[y_train == class_index] = n_occurr
    sample_weight = sample_weight / total_samples

    # ## Clustering

    # In[7]:

    n_leaves = 4
    n_clusters = n_leaves
    clustering_estimators = []
    params = dict(n_clusters=n_clusters, random_state=SEED)
    kmeans = KMeans(**params)
    clustering_estimators.append(kmeans)

    # Spectral clustering not used since it gave looped

    params = dict(n_clusters=n_clusters, linkage="single")
    agglomerate = AgglomerativeClustering(**params)
    clustering_estimators.append(agglomerate)

    params = dict(n_clusters=n_clusters, linkage="ward")
    agglomerate = AgglomerativeClustering(**params)
    clustering_estimators.append(agglomerate)

    params = dict(n_clusters=n_clusters, linkage="complete")
    agglomerate = AgglomerativeClustering(**params)
    clustering_estimators.append(agglomerate)

    params = dict(n_clusters=n_clusters, linkage="average")
    agglomerate = AgglomerativeClustering(**params)
    clustering_estimators.append(agglomerate)

    params = dict(n_clusters=n_clusters)
    birch = Birch(**params)
    clustering_estimators.append(birch)

    # In[8]:

    from src.cluster import find_best_estimator

    for i in range(len(clustering_estimators)):
        try:
            clustering_estimators[i] = clustering_estimators[i].fit(X_train, sample_weight=sample_weight.transpose())
        except:
            clustering_estimators[i] = clustering_estimators[i].fit(X_train)

    for estimator in clustering_estimators:
        print(estimator.__class__.__name__, homogeneity_score(y_train, estimator.labels_))

    best_estimator = find_best_estimator(clustering_estimators, homogeneity_score, y_train)
    print("The best estimator is {}".format(best_estimator))

    # ## Leaves assignment

    # In[9]:

    from src.cluster import best_leaf_assignment

    for estimator in clustering_estimators:
        assignment, score = best_leaf_assignment(n_leaves=n_leaves, estimated_labels=estimator.labels_,
                                                 true_labels=y_train, metric=completeness_score)
        print("For the estimator {}, the assignment {} has a score of {}".format(estimator.__class__.__name__,
                                                                                 assignment, score))

    # ## Parameters Initialization

    # In[10]:

    from src.cluster import HierarchicalLogisticRegression

    HLR = HierarchicalLogisticRegression(n_classes=len(np.unique(y_train)), n_leaves=n_leaves,
                                         prediction_type="deterministic", random_state=0,
                                         logistic_params={"class_weight": "balanced"})
                                                          # "penalty": "elasticnet",
                                                          # "solver": "saga",
                                                          # "l1_ratio": 0.9,
                                                          # "fit_intercept": False})

    # In[11]:

    best = clustering_estimators[0]
    best_accuracy = 0
    i = 0
    for estimator in clustering_estimators:
        """
        print(estimator)
        print(np.unique(estimator.labels_))
        for un in np.unique(estimator.labels_):
            print(len(estimator.labels_[estimator.labels_==un]))
        """
        assignment, score = best_leaf_assignment(n_leaves=n_leaves, estimated_labels=estimator.labels_,
                                                 true_labels=y_train, metric=completeness_score)
        HLR = HLR.fit(X_train.to_numpy(), y_train, cluster_labels=estimator.labels_, leaves_assignment=assignment)
        acc = HLR.score(X_test.to_numpy(), y_test)
        print("{} accuracy:{}".format(estimator, acc))
        if acc > best_accuracy:
            best = clustering_estimators[i]
            best_accuracy = acc
        i += 1
    print("\nThe best was {} with score {}".format(best, best_accuracy))

    # In[12]:

    assignment, score = best_leaf_assignment(n_leaves=n_leaves, estimated_labels=y_train,
                                             true_labels=y_train, metric=completeness_score)
    print("The true labelling has assignment {} with score {}".format(assignment, score))
    HLR = HierarchicalLogisticRegression(n_classes=len(np.unique(y_train)),
                                         n_leaves=n_leaves, prediction_type="deterministic",
                                         random_state=0)
    HLR = HLR.fit(X_train.to_numpy(), y_train, cluster_labels=y_train, leaves_assignment=assignment)
    acc = HLR.score(X_test.to_numpy(), y_test)
    print("Accuracy using true labellling: {}".format(acc))


    # In this notebook there is the only case in which a clustrering algorithm performs better than the true labelling.

    # ## ORCT model

    # In[13]:

    def B_in_NR(model, i):
        if i == 4:
            return []
        elif i == 5:
            return [2]
        elif i == 6:
            return [1]
        elif i == 7:
            return [1, 3]


    def B_in_NL(model, i):
        if i == 4:
            return [1, 2]
        elif i == 5:
            return [1]
        elif i == 6:
            return [3]
        elif i == 7:
            return []


    def I_k(model, i):
        if i == 0:
            return I_in_k[0]
        elif i == 1:
            return I_in_k[1]
        elif i == 2:
            return I_in_k[2]


    # In[14]:

    BF_in_NL_R = {4: [], 5: [2], 6: [1], 7: [1, 3]}
    BF_in_NL_L = {4: [1, 2], 5: [1], 6: [3], 7: []}
    I_in_k = {i: list(df_train[df_train['Classes'] == i].index) for i in range(len(classes_en))}
    my_W = {(i, j): 0.5 if i != j else 0 for i in classes_en for j in classes_en}
    index_instances = list(X_train.index)
    my_x = {(i, j): df_train.loc[i][j] for i in index_instances for j in index_features}

    # In[15]:

    a = np.stack(HLR.coef_).transpose() /512
    # mu = np.zeros_like(np.stack(HLR.intercept_))
    mu=np.stack(HLR.intercept_) /512
    C = HLR.leaf_class_probs_.transpose()
    # j+1 due to the convention for the branch nodes (numbered from 1)
    # it's in the form
    # (0,1) (0,2) (0,3)
    # (1,1) (1,2) (1,3) and so one
    init_a = {(i, j + 1): a[i, j] for i in range(len(index_features)) for j in range(3)}
    print("a\n", a)
    print("C\n", C)
    print("mu\n", mu)
    print(HLR.leaf_class_probs_)
    # in the form (1) (2) (3)
    init_mu = {(i + 1): mu[i] for i in range(3)}
    # shape (n_classes, n_leaves), and leaves are the last 4 numbers of 2^h -1
    # (0,4) (0,5) (0,6) (0,7)
    # (1,4) ---
    init_c = {(i, j + 4): C[i, j] for i in classes_en for j in range(4)}

    # In[16]:
    model = ORCTModel(I_in_k=I_in_k, I_k_fun=I_k, index_features=index_features, BF_in_NL_R=BF_in_NL_R,
                      B_in_NR=B_in_NR, B_in_NL=B_in_NL, error_weights=my_W, x_train=my_x, init_a=init_a,
                      init_mu=init_mu, init_C=init_c)

    ipopt_path = "~/miniconda3/envs/decision_trees/bin/ipopt"
    model.solve(ipopt_path)
    model.print_results()
    val = model.extraction_va()
    print(val["a"])
    labels = predicted_lab(model.model, X_test, val, index_features)
    a = accuracy(y_test.to_numpy(), labels)
    print("accuracy", a)
