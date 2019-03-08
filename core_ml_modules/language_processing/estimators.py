import re
from collections import Counter

import nltk

from sklearn.base import BaseEstimator
from sklearn.pipeline import TransformerMixin
from sklearn.feature_extraction import DictVectorizer

class Tokeniser(BaseEstimator, TransformerMixin):
    """
    Transformer object tokenising messages with a specific tokeniser
    Sci-kit learn documentation on creating estimators: http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
    """
    def fit(self, X, y=None):
        """
        Fit simply returns self, no other information is needed.
        """
        return self

    def inverse_transform(self, X):
        """
        No inverse transformation
        """
        return X

    def transform(self, messages):
        """
        Tokenises messages
        """
        tokenised_messages = []

        for message in messages:
            tokenised_messages.append(self.tokenise(message))

        return tokenised_messages

    def tokenise(self, input_string):
        """
        Tokenises a string, removing hyphens and replacing "&" with "and" in the process.

        :type input_string: str
        :return: List of tokens
        :rtype: list of str
        """
        output = []
        for m in re.finditer("[a-zA-Z0-9]+(\'[a-zA-Z])?|\(+|\)+|\?+|[^a-zA-Z0-9\s]+(\s+[^a-zA-Z0-9\s]+)*",
                             re.sub("-", "", input_string)):
            output.append(m.group())
        return output

class NGramFrequencyExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer object turning messages into frequency feature vectors counting ngrams up to specified maximum.
    Sci-kit learn documentation on creating estimators: http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
    """

    def __init__(self, lexicon, form=None, default_form=lambda word, lexicon: word, ngram_size=1, adjust_for_message_len=True):
        self.lexicon = lexicon
        self.form = form
        self.default_form = default_form
        self.ngram_size = ngram_size
        self.vectorizer = DictVectorizer()
        self.adjust_for_message_len = adjust_for_message_len

    def extract_frequency_dicts(self, X):
        frequency_dicts = []
        for message in X:
            tuple_ngrams = nltk.ngrams(self.retrieve_lexical_form(message), self.ngram_size)
            string_ngrams = []
            for ngram in tuple_ngrams:
                string_ngrams.append(",".join(ngram))

            frequency_dict = Counter(string_ngrams)
            if self.adjust_for_message_len:
                for ngram in frequency_dict:
                    frequency_dict[ngram] = frequency_dict[ngram] / len(string_ngrams)

            frequency_dicts.append(frequency_dict)

        return frequency_dicts

    def fit(self, X, y=None):
        """
        Determines the list of tokens and ngrams to be used
        :param X: List of tokenised messages
        :type X: list(list(str))
        """
        frequency_dicts = self.extract_frequency_dicts(X)
        self.vectorizer.fit(frequency_dicts)
        return self

    def transform(self, X, y=None):
        """
        Transforms tokenised messages into frequency vectors
        :return: frequency vectors
        :rtype: numpy array of shape [n_samples, n_features]
        """
        frequency_dicts = self.extract_frequency_dicts(X)
        return self.vectorizer.transform(frequency_dicts)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data then transform it
        :return: frequency vectors
        :rtype: numpy array of shape [n_samples, n_features]
        """
        frequency_dicts = self.extract_frequency_dicts(X)
        return self.vectorizer.fit_transform(frequency_dicts)

    def get_feature_names(self):
        try:
            return self.vectorizer.get_feature_names()
        except AttributeError:
            raise AttributeError("No feature names, object has not been fitted")

    def retrieve_lexical_form(self, message):
        if self.form is None:
            return message

        assert self.lexicon.has_feature(self.form)

        transformed_message = []
        for word in message:
            if word in self.lexicon and self.lexicon.get_feature_value_by_word(word, self.form):
                transformed_message.append(self.lexicon.get_feature_value_by_word(word, self.form))
            else:
                transformed_message.append(self.default_form(word, self.lexicon))

        return transformed_message

class Debugger(BaseEstimator, TransformerMixin):
    def __init__(self, print_string):
        self.print_string = print_string

    def transform(self, X):
        print(X.shape, self.print_string)
        return X

    def fit(self, X, y=None, **fit_params):
        print(len(y), self.print_string)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        print(X.shape, len(y), self.print_string)
        return X

