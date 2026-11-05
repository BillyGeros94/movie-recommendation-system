# Hyperparameters chosen on the validation split by tune.py. The test set plays no part in choosing them

UV_INIT = {'n_factors': 50, 'std': 0.05, 'fold_in_reg': 2.0}
UV_FIT = {'n_epochs': 43, 'lr': 0.01, 'reg': 0.05, 'batch_size': 20000}
ITEM_CF = {'n_neighbors': 200, 'sim_shrinkage': 10.0, 'weight_shrinkage': 0.1}
USER_CF = {'n_neighbors': 50, 'sim_shrinkage': 50.0, 'weight_shrinkage': 0.1}
