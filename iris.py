# dataframe management
import os.path
import time

import numpy as np
import pandas as pd
from pyomo.environ import *
from sklearn import datasets
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import completeness_score, homogeneity_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sorct import SORCT
from src.cluster import HierarchicalLogisticRegression, best_leaf_assignment
from src.utils import get_number_of_iterations
from sklearn.model_selection import KFold
from src.cluster import find_best_estimator

def create_model(dataset_name, df_train, X_test, y_test, classes, random_init, HLR=None, opt_tipe="simple",
                 ipopt_path="~/miniconda3/envs/decision_trees/bin/ipopt", tee=False):
    I_in_k_in = {i: list(df_train[df_train['Classes'] == i].index) for i in range(len(classes))}
    index_instances = list(df_train.index)
    init_vals = []
    model = SORCT(dataset=df_train, I_in_k=I_in_k_in, I_k=I_in_k_in)
    model.createModel()
    model.charge_of(opt_tipe)
    if random_init:
        assert HLR is None
    if not random_init:
        params = HLR.get_ORCT_params()
        a = params["a"]
        mu = params["mu"]
        C = params["C"]
        # j+1 due to the convention for the branch nodes (numbered from 1)
        # it's in the form
        # (0,1) (0,2) (0,3)
        # (1,1) (1,2) (1,3) and so one
        init_a = {(i, j + 1): a[i, j] for i in range(len(index_features)) for j in range(3)}
        # in the form (1) (2) (3)
        init_mu = {(i + 1): mu[i] for i in range(3)}
        # shape (n_classes, n_leaves), and leaves are the last 4 numbers of 2^h -1
        # (0,4) (0,5) (0,6) (0,7)
        # (1,4) ---
        init_c = {(i, j + 4): C[i, j] for i in classes_en for j in range(4)}
        Pr = HLR.leaves_probabilities(X_train.to_numpy())
        prob_df = pd.DataFrame(Pr, index=df_train.index)
        init_Pr = {(i, j + 4): prob_df.loc[i][j] for i in index_instances for j in range(4)}
        init_vals.extend([init_a, init_mu, init_c, init_Pr])
        model.set_init(init_vals)
    try:
        # try catch for when max number of iteration is met
        results, solver = model.solve(ipopt_path, tee=TEE_VALUE)
    except:
        return -1, -1, -1, None
    sorct_time = results.solver.time
    sorct_term_cond = results.solver.termination_condition
    assert_optimal_termination(results)
    stringa = solver.__dict__["_log"]
    sorct_iters = get_number_of_iterations(stringa)
    # model.model.display()
    model.extraction_va()
    pred_labels = model.predicted_lab(X_test)
    if dataset_name == "new_thyroid":
        sorct_score = balanced_accuracy_score(y_test, pred_labels)
    else:
        sorct_score = model.accuracy(y_test, pred_labels)

    return sorct_time, sorct_iters, sorct_score, sorct_term_cond


def fit_HLR(X_train, y_train, n_leaves=4, random_state=None, use_true_labels=True, estimator=None, balanced=False):
    logistic_params = {}
    if balanced:
        logistic_params = {"class_weight": "balanced"}
    if use_true_labels:
        assert estimator is None
        labels = y_train
    else:
        assert estimator is not None
        labels = estimator.labels_
    HLR = HierarchicalLogisticRegression(n_classes=len(np.unique(y_train)), n_leaves=n_leaves,
                                         prediction_type="deterministic", random_state=SEED,
                                         logistic_params=logistic_params)
    assignment, score = best_leaf_assignment(n_leaves=n_leaves, estimated_labels=labels,
                                             true_labels=y_train, metric=completeness_score)
    HLR = HLR.fit(X_train.to_numpy(), y_train, cluster_labels=labels, leaves_assignment=assignment)
    return HLR


def create_clusters(n_leaves, SEED):
    n_clusters = n_leaves
    clustering_estimators = []
    names = []
    params = dict(n_clusters=n_clusters, random_state=SEED)
    kmeans = KMeans(**params)
    clustering_estimators.append(kmeans)
    names.append("kmeans")

    # Spectral clustering not used since it gave looped

    params = dict(n_clusters=n_clusters, linkage="single")
    agglomerate = AgglomerativeClustering(**params)
    clustering_estimators.append(agglomerate)
    names.append("Agglomerative_sigle")

    # params = dict(n_clusters=n_clusters, linkage="ward")
    # agglomerate = AgglomerativeClustering(**params)
    # clustering_estimators.append(agglomerate)
    # names.append("Agglomerative_ward")
    #
    # params = dict(n_clusters=n_clusters, linkage="complete")
    # agglomerate = AgglomerativeClustering(**params)
    # clustering_estimators.append(agglomerate)
    # names.append("Agglomerative_complete")
    #
    # params = dict(n_clusters=n_clusters, linkage="average")
    # agglomerate = AgglomerativeClustering(**params)
    # clustering_estimators.append(agglomerate)
    # names.append("Agglomerative_average")

    params = dict(n_clusters=n_clusters)
    birch = Birch(**params)
    clustering_estimators.append(birch)
    names.append("birch")
    return clustering_estimators, names


if __name__ == "__main__":


    import logging

    logging.getLogger('pyomo.core').setLevel(logging.ERROR)
    ALL_START = time.time()
    N_SPLITS = 5
    OPT_TYPE = "simple"
    SEED = 1234
    np.random.random(SEED)
    TEE_VALUE = False
    dataset_name_list = ["car", "iris", "new_thyroid", "seeds_data", "splice"]
    for dataset_name in dataset_name_list:
        if dataset_name == "iris":
            X, y = datasets.load_iris(as_frame=True, return_X_y=True)
            df = pd.DataFrame(X)
            df["Classes"] = y
        elif dataset_name == "car":
            dataset_path = os.path.join("datasets", "{}.csv".format(dataset_name))
            df = pd.read_csv(dataset_path, delimiter=";", header=0)
            df = df.convert_dtypes()
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
        else:
            dataset_path = os.path.join("datasets", "{}.csv".format(dataset_name))
            df = pd.read_csv(dataset_path, delimiter=";", header=0)
        if "Id" in df:
            df = df.drop('Id', axis=1)
        df_std = df.copy()
        scaler = MinMaxScaler()  # also MaxAbsScaler()
        # Preprocessing: we get the columns names of features which have to be standardized
        columns_names = list(df)
        index_features = list(range(0, len(df_std.columns) - 1))
        # The name of the classes K
        classes = df_std['Classes'].unique().tolist()
        classes_en = [i for i in range(len(classes))]
        # Encoder processing
        le = preprocessing.LabelEncoder()
        le.fit(df_std['Classes'])
        df_std['Classes'] = le.transform(df_std['Classes'])
        # Scaling phase
        df_std[columns_names[0:-1]] = scaler.fit_transform(df_std[columns_names[0:-1]])

        X = df_std[columns_names[:-1]]
        y = df_std[columns_names[-1]]
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        n_leaves = 4

        _, names = create_clusters(4, SEED)
        names.append("True_labels")
        df_columns = []
        for col_idx in range(N_SPLITS):
            df_columns.append("Time_{}".format(col_idx))
            df_columns.append("HLR_Time_{}".format(col_idx))
            df_columns.append("Iterations_{}".format(col_idx))
            df_columns.append("HLR_Score_{}".format(col_idx))
            df_columns.append("SORCT_Score_{}".format(col_idx))
            df_columns.append("Homogeneity_{}".format(col_idx))
            df_columns.append("Completeness_{}".format(col_idx))
        clustering_df = pd.DataFrame(index=names, columns=df_columns)
        sorct_columns = []
        for sorct_col_idx in range(N_SPLITS):
            df_columns.append("Time_{}".format(sorct_col_idx))
            df_columns.append("Iterations_{}".format(sorct_col_idx))
            df_columns.append("Score_{}".format(sorct_col_idx))
        sorct_df = pd.DataFrame(columns=sorct_columns)

        fold_index = 0
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            df_train = pd.concat([X_train, y_train], axis=1)
            df_test = pd.concat([X_test, y_test], axis=1)
            # sample weighting
            occurences = [len(y_train[y_train == x]) for x in classes]
            total_samples = sum(occurences)
            sample_weight = np.zeros_like(y_train)
            for class_index, n_occurr in zip(classes, occurences):
                sample_weight[y_train == class_index] = n_occurr
            sample_weight = sample_weight / total_samples


            # test no init sorct
            sorct_time, sorct_iters, sorct_score, sorct_term_cond = \
                create_model(dataset_name,df_train, X_test, y_test, classes, random_init=True, opt_tipe=OPT_TYPE, tee=TEE_VALUE)
            sorct_df.loc["SORCT", "Time_{}".format(fold_index)] = sorct_time
            sorct_df.loc["SORCT", "Iterations_{}".format(fold_index)] = sorct_iters

            sorct_df.loc["SORCT", "Score_{}".format(fold_index)] = sorct_score
            # true label performances
            start = time.time()
            HLR = fit_HLR(X_train, y_train, n_leaves=4, random_state=SEED, use_true_labels=True, balanced=True)
            end = time.time()
            HLR_score_tl = HLR.score(X_test.to_numpy(), y_test)
            clustering_df.loc["True_labels", "HLR_Time_{}".format(fold_index)] = end - start
            clustering_df.loc["True_labels", "HLR_Score_{}".format(fold_index)] = HLR_score_tl
            sorct_time, sorct_iters, sorct_score, sorct_term_cond = \
                create_model(dataset_name, df_train, X_test, y_test, classes, tee=TEE_VALUE,random_init=False, HLR=HLR, opt_tipe=OPT_TYPE)
            clustering_df.loc["True_labels", "Time_{}".format(fold_index)] = sorct_time
            clustering_df.loc["True_labels", "Iterations_{}".format(fold_index)] = sorct_iters
            clustering_df.loc["True_labels", "SORCT_Score_{}".format(fold_index)] = sorct_score
            clustering_df.loc["True_labels", "Homogeneity_{}".format(fold_index)] = 1
            clustering_df.loc["True_labels", "Completeness_{}".format(fold_index)] = 1
            # clustering estimators
            clustering_estimators, names = create_clusters(n_leaves=4, SEED=SEED)
            for cl_idx in range(len(clustering_estimators)):
            # for cl_idx in range(1):
                ce = clustering_estimators[cl_idx]
                cluster_name = names[cl_idx]
                try:
                    ce = ce.fit(X_train, sample_weight=sample_weight)
                except:
                    ce = ce.fit(X_train)
                hs = homogeneity_score(y_train, ce.labels_)
                cs = completeness_score(y_train, ce.labels_)
                clustering_df.loc[cluster_name, "Homogeneity_{}".format(fold_index)] = hs
                clustering_df.loc[cluster_name, "Completeness_{}".format(fold_index)] = cs
                cl_start = time.time()
                HLR = fit_HLR(X_train, y_train, n_leaves=4, random_state=SEED, use_true_labels=True, balanced=True)
                cl_end = time.time()
                HLR_score_cl = HLR.score(X_test.to_numpy(), y_test)
                clustering_df.loc[cluster_name, "HLR_Time_{}".format(fold_index)] = cl_end - cl_start
                clustering_df.loc[cluster_name, "HLR_Score_{}".format(fold_index)] = HLR_score_cl
                sorct_time, sorct_iters, sorct_score, sorct_term_cond = \
                    create_model(dataset_name, df_train, X_test, y_test, classes, random_init=False, HLR=HLR, opt_tipe=OPT_TYPE, tee=TEE_VALUE)
                clustering_df.loc[cluster_name, "Time_{}".format(fold_index)] = sorct_time
                clustering_df.loc[cluster_name, "Iterations_{}".format(fold_index)] = sorct_iters
                clustering_df.loc[cluster_name, "SORCT_Score_{}".format(fold_index)] = sorct_score

            fold_index += 1

        print(clustering_df)
        print(sorct_df)
        clustering_df.to_csv("results/{}_results.csv".format(dataset_name), sep=" ")
        sorct_df.to_csv("results/{}_sorct.csv".format(dataset_name), sep=" ")
    ALL_END = time.time()
    print("Time elapsed: {}".format(ALL_END - ALL_START))
