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
    verbose = False

    for word_id in test_set.get_all_Xlengths().keys():
        X, lengths = test_set.get_all_Xlengths()[word_id]

        subprobabilities = {}
        maxLogL = float("-inf")
        guess = None

        for word, model in models.items():
            try:
                logL = model.score(X, lengths)

                subprobabilities[word] = logL

                if logL > maxLogL:
                    maxLogL = logL
                    guess = word

            except Exception as e:
                if verbose:
                    print("failure on model word {}. Err: {}".format(word, str(e)))

        probabilities.append(subprobabilities)
        guesses.append(guess)


    return (probabilities, guesses)
