from pomegranate import *
import numpy as np

# Dearest Panika,
# Read Digit Model and then HMMEstimator, most of it is copied from their hmm.py script


# This class trains an HMM model for one digit, 10 of these will be made
class DigitModel:
    def __init__(self):
        self.model = None

    def fit(self, X, trans_mat, starts, data, n_states=2, n_mixtures=2, max_iterations=10):
        dists = []  # list of probability distributions for the HMM states
        for i in range(n_states):
            if n_mixtures > 1:
                a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_mixtures, X)
            else:
                a = MultivariateGaussianDistribution.from_samples(X)
            dists.append(a)

        model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, state_names=[f's{i}' for i in range(n_states)])
        model.fit(data, max_iterations=max_iterations)

        self.model = model

    # "Predict" in more of a binary classification sense
    def predict(self, X_test):
        logp, _ = self.model.viterbi(X_test)  # Run viterbi algorithm and return log-probability
        return logp


# Boring function to hide these lines of code, initializing as instructed
def initialize_parameters(n_states):
    trans_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        e = 1e-5
        p = np.random.uniform(0 + e, 1)  # Getting zero probabilities might not be desired
        if i + 1 < n_states:
            trans_matrix[i][i + 1] = p
            trans_matrix[i][i] = 1 - p
        else:
            trans_matrix[i][i] = 1

    starts = np.zeros(n_states)
    starts[0] = 1

    return trans_matrix, starts


# Wrapper class for the overall estimator
class HMMEstimator:
    def __init__(self, n_states, n_mixtures):
        self.models = []    # Will contain the 10 models

        # Hyperparameters
        self.n_states = n_states
        self.n_mixtures = n_mixtures

    def fit(self, X_train=None, y_train=None):
        for digit in range(10):
            # Create Data Array
            digit_train = X_train[y_train == digit]
            X = digit_train[0]
            for i in range(1, digit_train.shape[0]):
                X = np.concatenate((X, X_train[i]), axis=0)

            # Create and initialize parameters
            trans_matrix, starts = initialize_parameters(self.n_states)

            # Train model for this digit
            model = DigitModel()
            model.fit(X=X, trans_mat=trans_matrix, starts=starts, data=list(digit_train), n_states=self.n_states,
                      n_mixtures=self.n_mixtures)
            self.models.append(model)

    def predict(self, X_test):
        y_pred = []
        for sample in X_test:
            likelihoods = []
            for model in self.models:
                likelihoods.append(model.predict(sample))

            y_pred.append(np.argmax(likelihoods))
        return np.array(y_pred)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_test == y_pred), y_pred
