# dataframe management
import pandas as pd
import math
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, completeness_score
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from functools import reduce  # Valid in Python 2.6+, required in Python 3
import operator
from pyomo.environ import *
from pyomo.opt import SolverFactory
from sklearn import datasets

from src.ORCTModel import ORCTModel, predicted_lab, accuracy
from src.cluster import HierarchicalLogisticRegression, best_leaf_assignment


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


if __name__ == "__main__":
    X, y = datasets.load_iris(as_frame=True, return_X_y=True)
    iris = pd.DataFrame(X)
    iris["Species"] = y
    # iris = iris.drop('Id', axis=1)
    iris_std = iris.copy()
    iris.head(5)
    scaler = MinMaxScaler()  # also MaxAbsScaler()
    # Preprocessing: we get the columns names of features which have to be standardized
    columns_names = list(iris)
    index_features = list(range(0, len(iris_std.columns) - 1))

    # The name of the classes K
    classes = iris_std['Species'].unique().tolist()
    classes_en = [i for i in range(len(classes))]

    # Encoder processing
    le = preprocessing.LabelEncoder()
    le.fit(iris_std['Species'])

    iris_std['Species'] = le.transform(iris_std['Species'])

    # Scaling phase
    iris_std[columns_names[0:4]] = scaler.fit_transform(iris_std[columns_names[0:4]])

    iris_std.head(1)

    df = iris_std[columns_names[:-1]]
    y = iris_std[columns_names[4]]
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_train.head(5)

    # training part
    n_leaves = 4
    HLR = HierarchicalLogisticRegression(n_classes=len(np.unique(y)), n_leaves=n_leaves, random_state=0,
                                         logistic_params={"class_weight": "balanced"})
    true_values = y_train
    ass, score = best_leaf_assignment(n_leaves, true_values, true_values, completeness_score)
    HLR = HLR.fit(X_train, y_train, true_values, ass)

    print("leaf_classes", HLR.leaf_classes_)
    print("leaf classe probs\n", HLR.leaf_class_probs_)
    print("cluster leaves association: ", HLR.cluster_leaves_association_)

    BF_in_NL_R = {4: [], 5: [2], 6: [1], 7: [1, 3]}
    BF_in_NL_L = {4: [1, 2], 5: [1], 6: [3], 7: []}
    I_in_k = {i: list(df_train[df_train['Species'] == i].index) for i in range(len(classes))}
    my_W = {(i, j): 0.5 if i != j else 0 for i in classes_en for j in classes_en}
    index_instances = list(X_train.index)
    my_x = {(i, j): df_train.loc[i][j] for i in index_instances for j in index_features}

    a = np.stack(HLR.coef_).transpose()
    mu = np.stack(HLR.intercept_)
    C = HLR.leaf_class_probs_.transpose()
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

    model = ORCTModel(I_in_k=I_in_k, I_k_fun=I_k, index_features=index_features, BF_in_NL_R=BF_in_NL_R,
                      B_in_NR=B_in_NR, B_in_NL=B_in_NL, error_weights=my_W, x_train=my_x, init_a=init_a,
                      init_mu=init_mu, init_C=init_c)

    ipopt_path = "~/miniconda3/envs/decision_trees/bin/ipopt"
    model.solve(ipopt_path)

    val = model.extraction_va()


    labels = predicted_lab(model.model, X_test, val, index_feature)
    a = accuracy(y_test.to_numpy(), labels)


    init_a = np.random.uniform(0,1,None)
    init_c = np.random.uniform(0,1,None)
    init_mu = np.random.uniform(0,1,None)
    model_no_init = ORCTModel(I_in_k=I_in_k, I_k_fun=I_k, index_features=index_features, BF_in_NL_R=BF_in_NL_R,
                      B_in_NR=B_in_NR, B_in_NL=B_in_NL, error_weights=my_W, x_train=my_x, init_a=init_a,
                      init_mu=init_mu, init_C=init_c)
    model_no_init.solve(ipopt_path)
    val_no_init = model_no_init.extraction_va()
    labels_no_init = predicted_lab(model_no_init.model, X_test, val_no_init, index_features)
    a_no_init = accuracy(y_test.to_numpy(), labels_no_init)
    print("\n\n\n")
    model.print_results()
    model_no_init.print_results()
    print("HLR=", HLR.score(X_test.to_numpy(), y_test.to_numpy()))
    print("ORCT=", a)
    print("ORCT no init", a_no_init)

