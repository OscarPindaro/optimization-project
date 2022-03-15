# optimization-project
Project of the OPTIMIZATION course. The objective is to find a good initialization for the random decision trees, in order to speed up convergence.

The algorithm used to divide handle the clustering of the dataset and the initialization are all in the "scr/cluster.py" file.

src/ORCTModel.py it's just used in the notebooks, but it was not used to train the agent. Instead,
the "sorct.py" file contains the correct model.

"source/" contains the utils used in sorct.py. This structure is inherited from the original code base.

run_tests.py loads all datasets, runs 4 different clustering algorithms, fit the HLR and then
runs the optimization step. The results are put in the results/ folder (create it if not present).
train_test.py loads a configuration of parameters and tests them on the folds of a train set. This folds are generated through
KMeans. While the code works properly, the predictions are all random and it's not
clear why.