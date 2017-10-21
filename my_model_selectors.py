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

    # p = model-df = number of parameters
    # N = num_states


    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        BIC = float("inf")
        model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states)
                if hmm_model != None:
                    logL = hmm_model.score(self.X, self.lengths)

                    """
                    p is the sum of four terms:
                        The free transition probability parameters, which is the size of the transmat matrix
                        The free starting probabilities
                        Number of means
                        Number of covariances which is the size of the covars matrix
                    """
                    p = (hmm_model.startprob_.size - 1) + (hmm_model.transmat_.size - 1) + hmm_model.means_.size + hmm_model.covars_.diagonal().size

                    N = self.X.shape[0]
                    tmpBIC =  -2 * logL + p * np.log(N)

                    if tmpBIC < BIC:
                        BIC = tmpBIC
                        model = hmm_model
            
            except Exception as e:
              if self.verbose:
                  print("failure on {} with {} states. Err: {}".format(self.this_word, num_states, str(e)))

        return model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_dic = float("-inf")
        best_model = None

        likelihoods = {}

        n_components = range(self.min_n_components, self.max_n_components + 1)
        models = {}

        for num_states in n_components:
            try:
                model = self.base_model(num_states)
                if model != None:
                    logL = model.score(self.X, self.lengths)
                    penalty = np.mean( [ model.score(self.hwords[word][0], self.hwords[word][1]) for word in self.words if word != self.this_word ] )
                    dic = logL - penalty

                    if dic > best_dic:
                          best_model = model
                          best_dic = dic
            except Exception as e:
                if self.verbose:
                    print("failure on {} with {} states. Err: {}".format(self.this_word, num_states, str(e)))
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        avgLogL = float("-inf")
        model = None

        best_n = 0
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            N = min([len(self.sequences), num_states])

            try:
                split_method = KFold(n_splits=N)
            except Exception:
                continue

            sumLogL = .0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                try:
                    trainX, trainLengths = combine_sequences(cv_train_idx, self.sequences)
                    testX, testLengths = combine_sequences(cv_test_idx, self.sequences)

                    hmm_model = GaussianHMM(n_components=N, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(trainX, trainLengths)
                    if hmm_model != None:
                        sumLogL += hmm_model.score(testX, testLengths)

                except Exception as e:
                    if self.verbose:
                        print("failure on {} with {} states. Err: {}".format(self.this_word, N, str(e)))

            if sumLogL / N > avgLogL:
                avgLogL = sumLogL / N
                best_n = N

        return self.base_model(best_n)
