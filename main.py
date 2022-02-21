import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, Birch
from sklearn.metrics import completeness_score, homogeneity_score
from sklearn.preprocessing import StandardScaler

from src.cluster import find_best_estimator, best_leaf_assignment, HierarchicalLogisticRegression

if __name__ == "__main__":
    DATASET_PATH = os.path.join("datasets", "car.csv")
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
    X= StandardScaler().fit_transform(X)
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
    print("true value", )
    print(estimator.labels_)
    n_leaves = 4
    HLR = HierarchicalLogisticRegression(n_classes=len(np.unique(y)), n_leaves=n_leaves, random_state=0)
    ass, score = best_leaf_assignment(n_leaves, estimator.labels_, true_values, homogeneity_score)
    best_leaf_assignment(4, true_values, true_values, completeness_score)
    HLR.fit(X, y, true_values, ass)
    print("leaf_classes", HLR.leaf_classes_)
    print("leaf classe probs\n", HLR.leaf_class_probs_)
    print("cluster leaves association: ", HLR.cluster_leaves_association_)
    for i in range(4):
        print("i", i, "n_samples:", len(y[y == i]))
    print("Coefficients\n", HLR.coef_)
    print("Intercepts\n", HLR.intercept_)

