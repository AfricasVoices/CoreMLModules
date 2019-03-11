import time
import re

import numpy as np

def log(str):
    """
    Prints provided string with timestamp
    """
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ": " + str)

def select_predicted_scores(scores, predicted_labels, classes):
    """
    Deals with different output shapes for binary and multi-label classification in scikit-learn.
    Returns scores as is if binary; returns scores for predicted labels if not binary.
    :param scores: Scores from predict_proba(), predict_log_proba(), or decision_function()
                   shape (n_samples,) if n_classes == 2 else (n_samples, n_classes)
    :type scores: np.array
    :param predicted_labels: Labels predicted by the classifier from predict()
                             shape [n_samples]
    :type predicted_labels: np.array
    :param classes: Class labels known to the classifier, from classes_
    :type classes: np.array
    :return: Scores for predicted labels, shape=(n_samples,)
    :rtype: np.array
    """
    if len(scores.shape) == 1:
        return scores

    predicted_scores = []
    for score_set, label in zip(scores, predicted_labels):
        score_index = list(classes).index(label)
        score = list(score_set)[score_index]
        predicted_scores.append(score)

    return np.array(predicted_scores)

def default_form_pos(word, lexicon):
    """
    Retrieves string that represents part of speech by default if there is no part of speech information in lexicon
    :type word: str
    :rtype: str
    """
    assert word, "Word cannot be empty string or None"

    if word.isnumeric():                        # Numbers (number length encoded in PoS tag)
        return "Num{}".format(len(word))
    elif re.fullmatch("[^a-zA-Z0-9]+", word):   # Punctuation
        return "Punct"
    else:                                       # Unknown
        return "Unk"
