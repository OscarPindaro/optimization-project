# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:16:01 2019

@author: antoc
"""
############                          DEPTH = 2

# Util.py is built in order to manage trees topologies & dataset architecture

import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.preprocessing import PolynomialFeatures

# Object to manage trees of depth 2

# dictionaries to deal with ancient nodes of leaf nodes
# Right
BF_in_NL_R = {4: [], 5: [2], 6: [1], 7: [1, 3]}
# Left
BF_in_NL_L = {4: [1, 2], 5: [1], 6: [3], 7: []}

# dictionaries to deal with ancient nodes of leaf nodes of decision tree of depth 1
BF_in_NL_R1 = {2: [], 3: [1]}
# Left
BF_in_NL_L1 = {2: [1], 3: []}


# By default the dataset is assumed as follows:
# - The column of the labels must be named "Classes"
# - The values that labels asummed are encoded by integer

# function to manage misclassification: return an dictionary of misclassificstion costs
def cost(dataset):
    # The name of the classes K
    classes = dataset['Classes'].unique().tolist()
    # Encoding
    classes_en = [i for i in range(len(classes))]

    return {(i, j): 0.5 if i != j else 0 for i in classes_en for j in classes_en}


# function to deal with indeces of instances among classes
def inst_class(dataset):
    # The name of the classes K
    classes = dataset['Classes'].unique().tolist()
    return {i: list(dataset[dataset['Classes'] == i].index) for i in range(len(classes))}


# function to deal with training set
# the input dataset must be a dataframe in pandas with all the column except for labels column
def my_train(dataset):
    index_instances = list(dataset.index)
    index_features = list(range(0, len(dataset.columns) - 1))

    return {(i, j): dataset.loc[i][j] for i in index_instances for j in index_features}


# function to manage ancient nodes of leaf nodes
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


def B_in_NR1(model, i):
    if i == 2:
        return []
    elif i == 3:
        return [1]


def B_in_NL1(model, i):
    if i == 2:
        return [1]
    if i == 3:
        return []


# function to manage the percentage of predictor variables per tree (Global regularization)
def lit(l):
    return 1 if 1 in l else 0


# function to get seeds values for random initialization
def generate_random_init(n_exp):
    init_data = list(np.random.randint(low=0, high=10000, size=n_exp))

    init_var = []
    for i in range(0, n_exp):
        var_i = list(np.random.randint(low=0, high=10000, size=9))
        init_var.append(var_i)
    return {'init_data': init_data, 'init_var': init_var}


# function to deal with random tree for random forest
def sampling_dataset(alpha, dataset, depth, init=[]):
    num_nodes = 2 ** (depth) - 1
    ini = init if init else list(np.random.randint(low=0, high=10000, size=num_nodes))

    x = dataset.iloc[:, :-1].copy()
    y = dataset.iloc[:, -1].copy()
    return {i + 1: pd.concat([x.sample(frac=alpha, replace=False, random_state=ini[i], axis=1), y], axis=1) for i in
            range(num_nodes)}


# function to expand the dimension of the dataset to a kernel of degree 2
def extended2_df(data_std):
    X = data_std.iloc[:, 0:data_std.shape[1] - 1]
    y = data_std.iloc[:, data_std.shape[1] - 1:data_std.shape[1]]
    num = X.shape[1]
    names = []
    names = [str(j) for j in range(num)]
    for j in range(num):
        for k in range(j, num):
            names.append(str(j) + '-' + str(k))
    poly = PolynomialFeatures(2, include_bias=False)
    New_X = poly.fit_transform(X)
    XX = pd.DataFrame(data=New_X)
    XX.columns = names
    data_std = pd.concat([XX, y], axis=1, join_axes=[XX.index])
    data_std = data_std.dropna()
    return data_std


# function to deal with prediction phase for random tree of random forest
def sampling_dataset1(alpha, dataset, depth, init=[]):
    num_nodes = 2 ** (depth) - 1
    ini = init if init else list(np.random.randint(low=0, high=10000, size=num_nodes))

    x = pd.DataFrame(dataset).T
    return {i + 1: x.sample(frac=alpha, replace=False, random_state=ini[i], axis=1) for i in range(num_nodes)}
