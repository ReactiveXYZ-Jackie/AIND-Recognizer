import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # loop through testing data
    for testing_word, (X, lengths) in test_set.get_all_Xlengths().items():
    	# logging current best choices
    	best_score = float('-inf')
    	best_guess = ""
    	probability_dict = {}
    	# loop through trained list to get the best choice
    	for trained_word, model in models.items():
    		try:
    			score = model.score(X, lengths)
    		except:
    			score = float('-inf')
    		# save the probability
    		probability_dict[trained_word] = score
    		# find best score
    		if score > best_score:
    			best_score = score
    			best_guess = trained_word
    	# save the best guess
    	guesses.append(best_guess)
    	# save probabilities
    	probabilities.append(probability_dict)

    return probabilities, guesses
