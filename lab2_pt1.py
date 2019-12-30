import numpy as np
from pr_lab2_2019_20_help_scripts.lab2_help.parser import parser
from HMMEstimator import HMMEstimator
import os
from sklearn.model_selection import train_test_split
import csv
import time
from matplotlib import pyplot as plt

start_time = time.time()

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
else:
    X_train_full = np.load(os.path.join(save_dir, 'X_train.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(save_dir, 'X_test.npy'), allow_pickle=True)
    y_train_full = np.load(os.path.join(save_dir, 'y_train.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(save_dir, 'y_test.npy'), allow_pickle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full)

# *** 2. Hyperparameter Optimization ***
optimize_parameters = True  # True to do the grid-search again

# Should we do a new stratified shuffle split for train and val sets every time?
# Should we do k-fold cross-validation for each set of hypes?
if optimize_parameters:
    scores = np.zeros((6, 7))
    for n_states in range(1, 5):
        for n_mixtures in range(1, 6):
            # Our own class, read it to get what's going on
            gmm_hmm = HMMEstimator(n_states, n_mixtures)
            gmm_hmm.fit(X_train, y_train)
            score, _ = gmm_hmm.score(X_val, y_val)
            scores[n_states][n_mixtures] = score

            print(
                f'Optimization, run:({n_states},{n_mixtures}), t:{time.time() - start_time}'
            )

    # argmax() returns the index in the flattened array
    idx = scores.argmax()

    # np.unravel_index(flat_index, ndarray_shape): Best function ever, takes a flat index and
    # returns the corresponding indices for an ndarray of given shape
    best_n_states, best_n_mixtures = np.unravel_index(idx, scores.shape)

    print(
        f'Best run: ({best_n_states}, {best_n_mixtures})'
    )

    with open(os.path.join(os.getcwd(), 'best_hyperparameters.txt'), 'w') as f:
        f.write(f'{best_n_states}, {best_n_mixtures}\n')

else:
    r = csv.reader(open(os.path.join(os.getcwd(), 'best_hyperparameters.txt')))
    best_n_states, best_n_mixtures = r.__next__()
    best_n_states, best_n_mixtures = int(best_n_states), int(best_n_mixtures)


# *** 3. Test the final model ***
gmm_hmm = HMMEstimator(best_n_states, best_n_mixtures)
gmm_hmm.fit(X_train_full, y_train_full)
final_score, y_pred = gmm_hmm.score(X_test, y_test)

print(
    f'My model brings all the boys to the yard\n'
    f'and damn right, it\'s better than yours:\n'
    f'Score: {final_score}\n'
    f'damn right, it\'s better than yours:\n'
    f'Score: {final_score}'
)


def plot_confusion_matrix(y_test, y_pred):
    cm = np.zeros((10, 10))
    for t, p in zip(y_test, y_pred):
        cm[t][p] += 1
    cm /= cm.max()

    plt.imshow(cm, cmap='Blues')
    plt.yticks(range(0, 10))
    plt.xticks(range(0, 10))
    plt.title('Confusion matrix')
    plt.show()


# Also do this for the validation set?
plot_confusion_matrix(y_test, y_pred)
