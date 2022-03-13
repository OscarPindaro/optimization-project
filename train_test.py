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

import run_tests
from sorct import SORCT
from src.cluster import HierarchicalLogisticRegression, best_leaf_assignment
from src.utils import get_number_of_iterations
from sklearn.model_selection import KFold
from src.cluster import find_best_estimator
from pyomo.opt import SolverStatus, TerminationCondition
import pickle


def get_model_init(dictionary, index_features, index_sample, classes_en):
    a = dictionary["a"]
    mu = dictionary["mu"]
    C = dictionary["C"]
    Pr = dictionary["Pr"]
    init_a = {(i, j + 1): a[i, j] for i in range(len(index_features)) for j in range(3)}
    # in the form (1) (2) (3)
    init_mu = {(i + 1): mu[i] for i in range(3)}
    # shape (n_classes, n_leaves), and leaves are the last 4 numbers of 2^h -1
    # (0,4) (0,5) (0,6) (0,7)
    # (1,4) ---
    init_c = {(i, j + 4): C[i, j] for i in classes_en for j in range(4)}
    init_pr = {(index_sample[i], j + 4): Pr[i, j] for i in range(len(index_sample)) for j in range(4)}
    return [init_a, init_mu, init_c, init_pr]


def create_model(init, opt_tipe, df_train, classes_en):
    I_in_k = {i: list(df_train[df_train['Classes'] == i].index) for i in range(len(classes_en))}
    model = SORCT(dataset=df_train, I_in_k=I_in_k, I_k=I_in_k)
    model.createModel()
    model.charge_of(opt_tipe)
    model.set_init(init)
    return model


def predict(model , X, y, dataset_name):
    model.extraction_va()
    pred_labels = model.predicted_lab(X)
    if dataset_name == "new_thyroid" or dataset_name == "car":
        sorct_score_f = balanced_accuracy_score(y, pred_labels)
    else:
        sorct_score_f = model.accuracy(y, pred_labels)
    return sorct_score_f

if __name__ == "__main__":
    # dataset_name = "splice"
    # dataset_path = os.path.join("datasets", "{}.csv".format(dataset_name))
    # df = pd.read_csv(dataset_path, delimiter=";", header=0)
    # df = df.convert_dtypes()
    # # dictionary converting ordinal categories to values
    # if dataset_name == "car":
    #     cost_dict = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
    #     doors_dict = {"2": 2, "3": 3, "4": 4, "5more": 5}
    #     persons_dict = {"2": 2, "4": 4, "more": 5}
    #     dimension_dict = {"small": 0, "med": 1, "big": 2}
    #     # buying
    #     df["buying"] = df["buying"].apply(lambda x: cost_dict[x])
    #     df["maint"] = df["maint"].apply(lambda x: cost_dict[x])
    #     df["doors"] = df["doors"].apply(lambda x: doors_dict[x])
    #     df["persons"] = df["persons"].apply(lambda x: persons_dict[x])
    #     df["lug_boot"] = df["lug_boot"].apply(lambda x: dimension_dict[x])
    #     df["safety"] = df["safety"].apply(lambda x: cost_dict[x])
    #     classes_encoder = preprocessing.LabelEncoder().fit(df["Classes"])
    #     df["Classes"] = classes_encoder.transform(df["Classes"])
    # N_SPLITS = 5
    # SEED = 1234
    # n_feature = len(df.columns) - 1
    # kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    #
    # scaler = MinMaxScaler()  # also MaxAbsScaler()
    # # Preprocessing: we get the columns names of features which have to be standardized
    # columns_names = list(df)
    # index_features = list(range(0, len(df.columns) - 1))
    # # The name of the classes K
    # classes = df['Classes'].unique().tolist()
    # classes_en = [i for i in range(len(classes))]
    # # Encoder processing
    # le = preprocessing.LabelEncoder()
    # le.fit(df['Classes'])
    # df['Classes'] = le.transform(df['Classes'])
    # # Scaling phase
    # df[columns_names[0:-1]] = scaler.fit_transform(df[columns_names[0:-1]])
    # for column in columns_names[0:-1]:
    #     # TODO janky solution to unreliable MinMaxScaler behaviour
    #     df.loc[df[column] > 1, column] = 1
    #     df.loc[df[column] < 0, column] = 0
    # X = df[df.columns[0:n_feature]]
    # y = df["Classes"]
    #
    # for train_index, test_index in kf.split(X, y):
    #     X_train, X_test = X.loc[train_index], X.loc[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     df_train = pd.concat([X_train, y_train], axis=1)
    #     df_test = pd.concat([X_test, y_test], axis=1)
    #     train_index = df_train.index
    #     break
    # filename = "HLR_tl_{}_0.pkl".format(dataset_name)
    # with open(os.path.join("results", filename), "rb") as f:
    #     hlr_dict = pickle.load(f)
    # OPT_TYPE = "simple"
    # init = get_model_init(hlr_dict, index_features, train_index, classes_en)
    # model = create_model(init, OPT_TYPE, df_train, classes_en)
    # hlr_score = predict(model, X_train, y_train, dataset_name)
    # print(hlr_score)
    # exit()




    dataset_name_list = ["car", "iris", "new_thyroid", "seeds_data", "splice"]
    N_SPLITS = 5
    OPT_TYPE = "simple"
    SEED = 1234
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
        for column in columns_names[0:-1]:
            # TODO janky solution to unreliable MinMaxScaler behaviour
            df_std.loc[df[column] > 1, column] = 1
            df_std.loc[df[column] < 0, column] = 0

        X = df_std[columns_names[:-1]]
        y = df_std[columns_names[-1]]
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        n_leaves = 4
        fold_index = 0

        hlr_clusters_scores = {}
        _, clusters_names = run_tests.create_clusters(4, SEED)
        sorct_no_init_scores = []
        sorct_cl_scores = {}

        for cl_name in clusters_names:
            hlr_clusters_scores[cl_name]=[]
            sorct_cl_scores[cl_name] = []
        hlr_clusters_scores["True_labels"] = []
        sorct_cl_scores["True_labels"] = []

        for train_index, test_index in kf.split(X, y):
            print("Fold", fold_index)
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            df_train = pd.concat([X_train, y_train], axis=1)
            df_test = pd.concat([X_test, y_test], axis=1)
            train_index = df_train.index
            # sample weightingsorct_iters
            occurences = [len(y_train[y_train == x]) for x in classes]
            total_samples = sum(occurences)
            sample_weight = np.zeros_like(y_train)
            for class_index, n_occurr in zip(classes, occurences):
                sample_weight[y_train == class_index] = n_occurr
            sample_weight = sample_weight / total_samples

            #******* HLR TRUE LABELS ****
            filename = "HLR_{}_{}_{}.pkl".format("tl", dataset_name, fold_index)
            with open(os.path.join("results", filename), "rb") as f:
                hlr_dict = pickle.load(f)
            hlr_init = get_model_init(hlr_dict, index_features, train_index, classes_en)
            model = create_model(hlr_init, OPT_TYPE, df_train, classes_en)
            hlr_score = predict(model, X_train, y_train, dataset_name)
            hlr_clusters_scores["True_labels"].append(hlr_score)
            # HLR_car_Agglomerative_sigle_0.pkl
            #*************** HLR TRUE CLUSTERING *************
            for cl_name in clusters_names:
                filename = "HLR_{}_{}_{}.pkl".format(dataset_name, cl_name, fold_index)
                with open(os.path.join("results", filename), "rb") as f:
                    cluster_hl_params = pickle.load(f)

                cluster_init = get_model_init(cluster_hl_params, index_features, train_index, classes_en)
                model = create_model(cluster_init, OPT_TYPE, df_train, classes_en)
                cluster_score = predict(model, X_train, y_train, dataset_name)
                hlr_clusters_scores[cl_name].append(cluster_score)
            #*********** END HLR ********************

            #******** SORCT no init
            #SORCT_no_init__car_0.pkl
            filename = "SORCT_no_init__{}_{}.pkl".format(dataset_name, fold_index)
            with open(os.path.join("results", filename), "rb") as f:
                sorct_params = pickle.load(f)
            sorct_init_vals = [sorct_params["a"], sorct_params["mu"], sorct_params["C"], sorct_params["P"]]
            model = create_model(sorct_init_vals, OPT_TYPE, df_train, classes_en)
            sorct_no_init_score = predict(model, X_train, y_train, dataset_name)
            sorct_no_init_scores.append(sorct_no_init_score)
            # *********** END SORCT no init **********

            #************ SORCT trained **************
            # SORCT true labelling
            # SORCT_tl_car_0.pkl
            filname = "SORCT_tl_{}_{}.pkl".format(dataset_name, fold_index)
            with open(os.path.join("results", filename), "rb") as f:
                sorct_tl_params = pickle.load(f)

            tl_sorct_init = [sorct_tl_params["a"], sorct_tl_params["mu"], sorct_tl_params["C"], sorct_tl_params["P"]]
            model = create_model(tl_sorct_init, OPT_TYPE, df_train, classes_en)
            sorct_tl_score = predict(model, X_train, y_train, dataset_name)
            sorct_cl_scores["True_labels"].append(sorct_tl_score)

            # SORCT_new_thyroid_birch_4.pkl
            for cl_name in clusters_names:
                filename = "SORCT_{}_{}_{}.pkl".format(dataset_name, cl_name, fold_index)
                with open(os.path.join("results", filename), "rb") as f:
                    cl_sorct_params = pickle.load(f)
                cl_sorct_int = [cl_sorct_params["a"], cl_sorct_params["mu"], cl_sorct_params["C"], cl_sorct_params["P"]]
                model = create_model(cl_sorct_int, OPT_TYPE, df_train, classes_en)
                sorct_cl_score = predict(model, X_train, y_train, dataset_name)
                sorct_cl_scores[cl_name].append(sorct_cl_score)
            # END SORCT CLUSTERING
            fold_index += 1

        # SAVE RESULTS
        df_filename = "result_train_{}.csv".format(dataset_name)
        result_df = pd.DataFrame(index=clusters_names)
        for cl_name in result_df.index:
            result_df.loc[cl_name, "HLR_Score"] = np.mean(hlr_clusters_scores[cl_name])
            result_df.loc[cl_name, "SORCT_Score"] = np.mean(sorct_cl_scores[cl_name])
        result_df.loc["True_labels", "HLR_Score"] = np.mean(hlr_clusters_scores["True_labels"])
        result_df.loc["True_labels", "SORCT_Score"] = np.mean(sorct_cl_scores["True_labels"])
        result_df.loc["SORCT", "SORCT_Score"] = np.mean(sorct_no_init_score)
        res_path = os.path.join("results", df_filename)
        print(result_df)
        result_df.to_csv(res_path, sep=" ")
        print("Dataframe of {} dataset saved".format(dataset_name))

