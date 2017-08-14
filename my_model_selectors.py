import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

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
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        train_results = []
        test_results = []

        for temp_n_split in range(self.min_n_components, min(self.max_n_components, len(self.sequences)) + 1):

            split_method = KFold(n_splits=temp_n_split, random_state=self.random_state)

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                # self.X = train_X
                # self.lengths = train_lengths
                # self.n_constant = temp_n_split

            try:
                hmm_model = GaussianHMM(n_components=temp_n_split, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(train_X, train_lengths)

                train_results.append(hmm_model.score(train_X, train_lengths))
                test_results.append(hmm_model.score(test_X, test_lengths))

                print("\nthis_word:", self.this_word)
                print("self.sequences:", len(self.sequences))
                print("temp_n_split:", temp_n_split)
                print("train sample score (logL)", hmm_model.score(train_X, train_lengths))
                print("test sample score (logL)", hmm_model.score(test_X, test_lengths))

                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, num_states))
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))

        np_train_results = np.array(train_results)
        np_test_results = np.array(test_results)

        print("\ntrain_results", train_results)
        print("np_train_results", np_train_results)
        print("train_results.argmax()", np_train_results.argmax())

        print("\ntest_results", test_results)
        print("Np_test_results", np_test_results)
        print("test_results.argmax()", np_test_results.argmax())


        best_num_components = self.n_constant
        return self.base_model(best_num_components)

        # return "TEST"

