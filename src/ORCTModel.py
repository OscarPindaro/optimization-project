import math

from pyomo.core import minimize, Objective, Constraint, value, exp
from pyomo.environ import ConcreteModel, PercentFraction, NonNegativeReals, Set, Param, Var, Reals
import numpy as np
from functools import reduce  # Valid in Python 2.6+, required in Python 3
import operator

from pyomo.opt import SolverFactory


class ORCTModel:
    """
    This was a class done by copying the content of a notebook of the orignal code base i was studying.
    """

    def __init__(self, I_in_k, I_k_fun, index_features, BF_in_NL_R, B_in_NR, B_in_NL, error_weights, x_train,
                 init_a, init_mu, init_C):
        # Instances & Classes
        # Assume a dict I_in_k, with keys k and values of a list of I's in that k
        model = ConcreteModel()  # ConcretModel()
        model.I = Set(initialize=set(i for k in I_in_k for i in I_in_k[k]))
        model.K = Set(initialize=I_in_k.keys())
        model.I_k = Set(model.K, initialize=I_k_fun)

        # Features
        model.f_s = Set(initialize=index_features)

        # Nodes Leaf N_L & Nodes Branch N_B
        model.N_B = Set(initialize=set(i for k in BF_in_NL_R for i in BF_in_NL_R[k]))
        model.N_L = Set(initialize=BF_in_NL_R.keys())
        model.N_L_R = Set(model.N_L, initialize=B_in_NR)
        model.N_L_L = Set(model.N_L, initialize=B_in_NL)
        # Cost of misclassification
        model.W = Param(model.K, model.K, within=NonNegativeReals, initialize=error_weights)

        # Value for the instance i-th of the feature j-th
        # model.x = Param(model.I, model.f_s, within=PercentFraction, initialize=x_train)
        model.x = Param(model.I, model.f_s, initialize=x_train)

        # Value for the lambda of global generalization
        model.lam_glob = Param(initialize=2)

        # random initialization
        init_beta = np.random.uniform(low=0.0, high=1.0, size=None)
        # TODO check if need to update these two porbabilities
        init_P = np.random.uniform(low=0.0, high=1.0, size=None)
        init_p = np.random.uniform(low=0.0, high=1.0, size=None)

        # The weigths of feature j-th in breanch node t-th
        model.a = Var(model.f_s, model.N_B, within=Reals, bounds=(-1,1), initialize=init_a)

        # auxiliary variables for smooth version of global regularization
        model.beta = Var(model.f_s, within=PercentFraction, initialize=init_beta)
        # The intercepts of the linear combinations correspond to decision variables
        model.mu = Var(model.N_B, within=Reals, bounds=(-1,1), initialize=init_mu)

        # The variables that take into account if node t is labeled with class k
        model.C = Var(model.K, model.N_L, within=PercentFraction, initialize=init_C)

        # An auxiliary variables
        model.P = Var(model.I, model.N_L, within=PercentFraction, initialize=init_P)
        model.p = Var(model.I, model.N_B, within=PercentFraction, initialize=init_p)

        model.cost = Objective(rule=cost_rule, sense=minimize)
        model.Pr = Constraint(model.I, model.N_L, rule=Pr)
        model.pr = Constraint(model.I, model.N_B, rule=pr)

        model.class_in_leaf = Constraint(model.N_L, rule=class_in_leaf)
        model.leaf_in_class = Constraint(model.K, rule=leaf_in_class)

        model.globalmin = Constraint(model.f_s, model.N_B, rule=global_min)
        model.globalma = Constraint(model.f_s, model.N_B, rule=global_ma)

        self.model = model
        self.results = None

    def solve(self, exec_path):
        self.results = None
        opt = SolverFactory('ipopt',
                            executable=exec_path)  # in executable the directory path of ipopt.exe
        opt.options["halt_on_ampl_error"] = "yes"
        results = opt.solve(self.model, tee=True)
        self.results = results
        return results

    def print_results(self):
        print(self.results)
        print(value(self.model.cost))

    # Function to store the variables results
    def extraction_va(self):
        model = self.model
        mu = {str(model.mu[i]): model.mu[i].value for i in model.mu}
        a = {str(model.a[i]): model.a[i].value for i in model.a}
        C = {str(model.C[i]): model.C[i].value for i in model.C}
        beta = {str(model.beta[i]): model.beta[i].value for i in model.beta}

        return {'mu': mu, 'a': a, 'C': C, 'beta': beta}


# Minimize the cost of misclassification
def cost_rule(model):
    return sum(sum(
        sum(model.P[i, t] * sum(model.W[k, j] * model.C[j, t] for j in model.K if k != j) for t in model.N_L) for i in
        model.I_k[k]) for k in model.K)
           #+ model.lam_glob * sum(model.beta[j] for j in model.f_s)


# We must add the following set of constraints for making a single class prediction at each leaf node:
def Pr(model, i, tl):
    return reduce(operator.mul, (model.p[i, t] for t in model.N_L_L[tl]), 1) * reduce(operator.mul,
                                                                                      (1 - model.p[i, tr] for tr in
                                                                                       model.N_L_R[tl]), 1) == model.P[
               i, tl]


def pr(model, i, tb):
    return 1 / (1 + exp(-512 * ((sum(model.x[i, j] * model.a[j, tb] for j in model.f_s) / 4) - model.mu[tb]))) == \
           model.p[i, tb]


# We must add the following set of constraints for making a single class prediction at each leaf node:
def class_in_leaf(model, tl):
    return sum(model.C[k, tl] for k in model.K) == 1


# We force each class k to be identified by, at least, one terminal node, by adding the set of constraints below:
def leaf_in_class(model, k):
    return sum(model.C[k, tl] for tl in model.N_L) >= 1


# The following set of constraints uanbles to manage global regularization
def global_min(model, f, tb):
    return model.beta[f] >= model.a[f, tb]


def global_ma(model, f, tb): \
        return model.beta[f] >= -model.a[f, tb]


def my_sigmoid(a, x, mu, scale=512):
    l = len(x)
    val = (sum([a[i] * x_vals for i, x_vals in enumerate(x)]) / l) - mu
    # The default value is 512 as suggested in Blanquero et Al.
    return 1 / (1 + math.exp(-scale * val))


# An easy way to manage product within elements of an iterable object
def multiply_numpy(iterable):
    return np.prod(np.array(iterable))


# Calculate the probability of an individual falling into a given leaf node:
def Prob(model, var, x, leaf_idx, index_features):
    left = [my_sigmoid(list(var['a']['a[' + str(i) + ',' + str(tl) + ']'] for i in index_features), x,
                       var['mu']['mu[' + str(tl) + ']']) for tl in model.N_L_L[leaf_idx]]
    right = [1 - my_sigmoid(list(var['a']['a[' + str(i) + ',' + str(tr) + ']'] for i in index_features), x,
                            var['mu']['mu[' + str(tr) + ']']) for tr in model.N_L_R[leaf_idx]]
    return multiply_numpy(left) * multiply_numpy(right)


# Calculate the predicted label of a single instance
def comp_label(model, x, var, index_features):
    prob = {
        k: sum(Prob(model, var, x, i, index_features) * var['C']['C[' + str(k) + ',' + str(i) + ']'] for i in model.N_L)
        for k in
        model.K}
    return int(max(prob, key=prob.get))


# Generate a list of predicted labels for the test set
def predicted_lab(model, X_test, var, index_features):
    label = []
    for i in range(0, len(X_test)):
        label.append(comp_label(model, list(X_test.iloc[i]), var, index_features))
    return label


# Calculate the accuracy out of sample
def accuracy(y, y_pred):
    l = [1 if y[i] == y_pred[i] else 0 for i in range(0, len(y))]
    return sum(l) / len(y)
