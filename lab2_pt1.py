import numpy as np
from pr_lab2_2019_20_help_scripts.lab2_help.parser import parser
from HMMEstimator import HMMEstimator
import os
from sklearn.model_selection import StratifiedShuffleSplit
import csv

read = False  # Change to re-read data
data_dir = os.path.join(os.getcwd(), 'fsdd-recordings')

# If this folder doesn't exist you might get an error, create it manually
save_dir = os.path.join(os.getcwd(), 'pickles')


# *** 1. Prepare Data ***
if read:
    data = parser(data_dir)
    X_train_full, X_test, y_train_full, y_test, spk_train, spk_test = parser(data_dir)
    np.save(arr=np.array(X_train_full), file=os.path.join(save_dir, 'X_train'))
    np.save(arr=np.array(X_test), file=os.path.join(save_dir, 'X_test'))
    np.save(arr=np.array(y_train_full), file=os.path.join(save_dir, 'y_train'))
    np.save(arr=np.array(y_test), file=os.path.join(save_dir, 'y_test'))
    np.save(arr=np.array(spk_train), file=os.path.join(save_dir, 'spk_train'))
    np.save(arr=np.array(spk_test), file=os.path.join(save_dir, 'spk_test'))
else:
    X_train_full = np.load(os.path.join(save_dir, 'X_train.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(save_dir, 'X_test.npy'), allow_pickle=True)
    y_train_full = np.load(os.path.join(save_dir, 'y_train.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'), allow_pickle=True)
    spk_train = np.load(os.path.join(save_dir, 'spk_train.npy'), allow_pickle=True)
    spk_test = np.load(os.path.join(save_dir, 'spk_test.npy'), allow_pickle=True)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42069)
train_idx, val_idx = sss.split(X_train_full, y_train_full).__next__()
X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]


# *** 2. Set HMM Parameters ***
# This code is actually useless here, it appears again as a function in the HMMEstimator module I made
# Probably TODO: delete
n_states = 5
trans_matrix = np.zeros((n_states, n_states))
for i in range(n_states):
    e = 1e-5
    p = np.random.uniform(0+e, 1)  # Getting zero probabilities might not be desired
    if i + 1 < n_states:
        trans_matrix[i][i+1] = p
        trans_matrix[i][i] = 1-p
    else:
        trans_matrix[i][i] = 1

starts = np.zeros(n_states)
starts[0] = 1  # Or maybe they mean starts[1]? Does it matter?


# *** 3. Train 10 GMM-HMM models ***
best_n_states = 2
best_n_mixtures = 2
optimize_parameters = True  # True to do the grid-search again

# Should we do a new stratified shuffle split for train and val sets every time?
# Should we do k-fold cross-validation for each set of hypes?
if optimize_parameters:
    scores = np.zeros((4, 5))
    for n_states in range(1, 5):
        for n_mixtures in range(1, 6):
            # Our own class, read it to get what's going on
            gmm_hmm = HMMEstimator(n_states, n_mixtures)
            HMMEstimator.fit(X_train, y_train)
            score = HMMEstimator.score(X_val, y_val)
            scores[n_states][n_mixtures] = score

    # argmax() returns the index in the flattened array
    idx = scores.argmax()

    # np.unravel_index(flat_index, ndarray_shape): Best function ever, takes a flat index and
    # returns the corresponding indices for an ndarray of given shape
    best_n_states, best_n_mixtures = np.unravel_index(idx, scores.shape)

    with open(os.path.join(os.getcwd(), 'best_hyperparameters.txt', 'wb')) as f:
        f.write(f'{best_n_states}, {best_n_mixtures}\n')
else:
    r = csv.reader(open(os.path.join(os.getcwd(), 'best_hyperparameters.txt')))
    best_n_states, best_n_mixtures = r.__next__()
    best_n_states, best_n_mixtures = int(best_n_states), int(best_n_mixtures)


# *** 4. Test the final model ***
gmm_hmm = HMMEstimator(best_n_states, best_n_mixtures)
gmm_hmm.fit(X_train_full, y_train_full)
final_score = gmm_hmm.score(X_test, y_test)

print(
    f'My model brings all the boys to the yard\n'
    f'and damn right, it\'s better than yours:\n'
    f'Score: {final_score}\n'
    f'damn right, it\'s better than yours\n'
    f'Score: {final_score}'
)


def confusion_matrix(self, y_test, y_pred):
    cf = np.zeros([y_pred.shape[0]] * 2)
    for t, p in y_test, y_pred:
        cf[t][p] += 1

    return cf
