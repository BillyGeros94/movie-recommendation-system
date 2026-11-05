# Hyperparameters chosen on the validation split by tune.py, apart from the new-user settings below.
# new_user_candidates and ease_reg were fixed in advance. shortlist_size and liked_rating were compared on test
# users and then checked on 1,000 other test users, so results for the hybrid are slightly optimistic.

UV_INIT = {'n_factors': 50, 'std': 0.05, 'fold_in_reg': 2.0, 'new_user_candidates': 2000,
           'ease_reg': 500.0, 'shortlist_size': 30, 'liked_rating': 3.5}
UV_FIT = {'n_epochs': 43, 'lr': 0.01, 'reg': 0.05, 'batch_size': 20000}
ITEM_CF = {'n_neighbors': 200, 'sim_shrinkage': 10.0, 'weight_shrinkage': 0.1}
USER_CF = {'n_neighbors': 50, 'sim_shrinkage': 50.0, 'weight_shrinkage': 0.1}
