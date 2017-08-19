import warnings
import numpy as np
from decimal import Decimal
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

    # TODO implement the recognizer
    # try:
    probabilities = []
    guesses = []

    # for key in models.keys():
    #     print(key)

    i = 0

    while i < test_set.num_items:
        word_prob_dict = {}
        best_model_name = "Test"
        best_model_prob = Decimal('-Infinity')
        word = test_set.wordlist[i]
        # print("\nAction word: ", word)

        X, lengths = test_set.get_item_Xlengths(i)
        # print("X", X)
        # print("lengths", lengths)

        for model_name, model_hmm in models.items():
            try:
                # print("Model tested:", model_name)
                word_score = model_hmm.score(X, lengths)
                # print("Model score:", word_score)
                word_prob_dict[model_name] = word_score
            except:
                word_prob_dict[model_name] = Decimal('-Infinity')

            if word_score >= best_model_prob:
                best_model_prob = word_score
                best_model_name = model_name

        probabilities.append(word_prob_dict)
        guesses.append(best_model_name)
        i += 1

    return probabilities, guesses

    # except:
    #     print("\nRecogniser Exception Raised")
    #     print("###########################")
    #     print()
    #     return None




