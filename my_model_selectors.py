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
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def bic_score(self, trial_num_components):
        """Helper to calculate BIC score
        
        BIC score for model with trial num of components
        
        Arguments:
            trial_num_components {Integer} -- Num of comps
        """
        # construct model based on number of comps
        model = self.base_model(trial_num_components)
        # get necessary properties
        logL = model.score(self.X, self.lengths)
        logN = math.log(len(self.X))
        # 
        # Why is p calculated in this manner ?
        # p = m^2 + km - 1 where m is number of number of states (n_components),
        # k is 2 since we are using Gaussian(normal) distribution
        # Source: https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
        # 
        p = self.n_components ** 2 + 2 * self.n_components - 1
        return -2 * logL + p * logN , model

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            # finding the best model
            best_score, model = max([ self.bic_score(n) for n in range(self.min_n_components, self.max_n_components + 1) ])
            return model
        except Exception as e:
            # fallback to default
            return self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def dic_score(self, trial_num_components):
        """Helper to calculate DIC score
        
        Calculating DIC score for the model with designated number of comps
        
        Arguments:
            trial_num_components {Integer} -- Number of comps
        """
        # construct model based on number of comps
        model = self.base_model(trial_num_components)
        # get necessary attributes
        logL = model.score(self.X, self.lengths)
        logP = [ model.score(X, lengths) for word, (X, lengths) in self.hwords.items() if word != self.this_word ]
        M = len(logP)
        return logL - sum(logP)/(M-1), model


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            # find best model
            best_score, model = max([ self.dic_score(n) for n in range(self.min_n_components, self.max_n_components + 1) ])
            return model
        except Exception as e:
            # fallback to default
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def cv_score(self, trial_num_components):
        """CV Average score calculator
        
        Helper to calculate cv score for a model with trial number of components
        
        Arguments:
            trial_num_components {Integer} -- Number of components
        """
        # init a score list for calculating average
        scores = []
        # number of splits response to sample size
        def num_splits():
            return math.floor(math.log(len(self.sequences), 2) + 1)
        # init KFold spliter 
        folded = KFold(n_splits = num_splits())
        # get trainning and testing subsets with the splitter acting on the current word seq
        for training_idx, testing_idx in folded.split(self.sequences):
            # combine the training subsets and save to mode
            self.X, self.lengths = combine_sequences(training_idx, self.sequences)
            # init a model based on above data
            model = self.base_model(trial_num_components)
            # combine the testing subsets
            X, lengths = combine_sequences(testing_idx, self.sequences)
            # reuse the model to check performance
            score = model.score(X, lengths)
            # save score
            scores.append(score)
        # find the average value for the score
        return np.mean(scores), model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        try:
            # select best score
            best_score, model = max([ self.cv_score(n) for n in range(self.min_n_components, self.max_n_components + 1) ])
            return model
        except:
            # fall back to default
            return self.base_model(self.n_constant)

