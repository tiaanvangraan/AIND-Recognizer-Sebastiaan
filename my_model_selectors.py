import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from sklearn.model_selection import train_test_split

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # TODO implement model selection based on BIC scores
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            train_results = []
            test_results = []
            hmm_n_nodes = self.min_n_components

            # for hmm_n_nodes in range(self.min_n_components, min(self.max_n_components, len(self.sequences)) + 1):
            while (hmm_n_nodes >= self.min_n_components) and (hmm_n_nodes <= min(self.max_n_components, len(self.sequences))):
                hmm_model = GaussianHMM(n_components=hmm_n_nodes, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False)

                hmm_model.fit(self.X, self.lengths)

                L = hmm_model.score(self.X, self.lengths)  # Log likelihood of fitted model
                N = len(self.X)  # Number of data points
                d = len(self.X[0])  # Number of features
                n = hmm_n_nodes  # Number of HMM states
                p = n*(n-1) + (n-1) + (2*d*n)  # Number of parameters

                train_results.append((-2*L) + (p*math.log(N)))
                test_results.append((-2*L) + (p*math.log(N)))
                hmm_n_nodes += 1

            np_hmm_n_nodes_results = np.array(test_results)
            best_num_components = self.min_n_components + np_hmm_n_nodes_results.argmin()

            return self.base_model(best_num_components)

        except:
            print("\nSelectorBIC Exception Raised")
            print("############################")
            print("this_word:", self.this_word)
            print("len(self.sequences)", len(self.sequences))
            print("min_n_components:", self.min_n_components)
            print("max_n_components", self.max_n_components)
            print()
            return None

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        # TODO implement model selection based on DIC scores
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            train_results = []
            test_results = []
            hmm_n_nodes = self.min_n_components

            while (hmm_n_nodes >= self.min_n_components) and (hmm_n_nodes <= min(self.max_n_components, len(self.sequences))):
                LPxii = []

                hmm_model = GaussianHMM(n_components=hmm_n_nodes, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False)

                hmm_model.fit(self.X, self.lengths)
                LPxi = hmm_model.score(self.X, self.lengths)  # Log likelihood of fitted model
                M = len(self.words.keys())

                for key in self.words.keys():
                    if key != self.this_word:
                        key_X, key_lengths = self.hwords[key]
                        LPxii.append(hmm_model.score(key_X, key_lengths))

                np_LPxii = np.array(LPxii)
                train_results.append(LPxi - (1 / ((M-1)*np_LPxii.sum())))
                hmm_n_nodes += 1

            np_train_results = np.array(train_results)
            best_num_components = self.min_n_components + np_train_results.argmax()

            return self.base_model(best_num_components)

        except:
            print("\nSelectorDIC Exception Raised")
            print("###########################")
            print("this_word:", self.this_word)
            print("len(self.sequences)", len(self.sequences))
            print("min_n_components:", self.min_n_components)
            print("max_n_components", self.max_n_components)
            print()
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):

        # TODO implement model selection using CV
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:

            split_method = KFold(n_splits=min(3, len(self.sequences)), random_state=self.random_state)
            hmm_n_nodes_results = []
            hmm_n_nodes = self.min_n_components

            # for hmm_n_nodes in range(self.min_n_components, min(self.max_n_components, len(self.sequences)) + 1):
            while (hmm_n_nodes >= self.min_n_components) and (hmm_n_nodes <= min(self.max_n_components, len(self.sequences))):
                train_results = []
                test_results = []
                hmm_model = GaussianHMM(n_components=hmm_n_nodes, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False)

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                    hmm_model.fit(train_X, train_lengths)
                    train_results.append(hmm_model.score(train_X, train_lengths))
                    test_results.append(hmm_model.score(test_X, test_lengths))

                np_train_results = np.array(train_results)
                np_test_results = np.array(test_results)
                hmm_n_nodes_results.append(np.mean(np_test_results))
                hmm_n_nodes += 1

            np_hmm_n_nodes_results = np.array(hmm_n_nodes_results)
            best_num_components = self.min_n_components + np_hmm_n_nodes_results.argmax()

            return self.base_model(best_num_components)

        except:
            print("\nSelectorCV Exception Raised")
            print("###########################")
            print("this_word:", self.this_word)
            print("len(self.sequences)", len(self.sequences))
            print("min_n_components:", self.min_n_components)
            print("max_n_components", self.max_n_components)
            print()
            return None


