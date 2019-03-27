class Lexicon:
    """
    Object containing lexical information on all words in the dataset
    :ivar _all_words: a list of all words in the dataset as Word objects
    :type _all_words: list(Word)
    :ivar _features: an index of linguistic features mapping onto words.
                     One outer dictionary maps the type of linguistic feature (e.g. "pos_tag") to the relevant inner dictionary,
                     and inner dictionaries map the feature value (e.g. "noun") to a list of Words that match the value
                     e.g. {
                               "pos_tag": {
                                   "noun": [Word("I"), Word("apples")],
                                   "verb": [Word("eat")]
                               },
                               "original_str": {
                                   "I": [Word("I")],
                                   "eat": [Word("eat")],
                                   "apples": [Word("apples")]
                               }
                           }
    :type _features: dict[str, dict[str, list(str)]]
    """

    class _Word:
        """
        Word object containing its original form and messages which contain it, as well as any other defined linguistic features
        :ivar _original_str: original form of the word
        :type _original_str: str
        :ivar _features: dictionary of the word's linguistic features and their values
        :type _features: dict(str)
        :ivar _messages: messages in which the word appears
        :type _messages: list(list(str))
        """
        def __init__(self, original_str, features):     # _Word constructor
            self._original_str = original_str
            self._features = dict.fromkeys(features, None)
            self._messages = []

        def get_original_str(self):
            return self._original_str

        def get_feature_value(self, feature):
            return self._features[feature]

        def set_feature_value(self, feature, value):
            self._features[feature] = value

        def get_messages(self):
            return self._messages

        def add_message(self, message):
            self._messages.append(message)

    def __init__(self, messages, features):       # Lexicon constructor
        assert isinstance(messages, list), "Messages must be in a list of lists of str representing tokens"
        assert isinstance(features, list), "Features must be in a list of str"

        self._all_words = {}
        self._features = dict.fromkeys(features, {})

        self._extract_words(messages)

    def __contains__(self, item):
        return item in self._all_words

    def _extract_words(self, messages):
        """
        Sorts words into a dictionary mapping original string to the corresponding Word object
        :param messages: list of tokenised messages
        :type messages: list(list(str))
        """
        for message in messages:
            assert isinstance(message, list), "Messages must be tokenised as a list of str"

            for word in message:
                assert isinstance(word, str), "Words must be represented as str"

                if word not in self._all_words:
                    self.add_word(word)

                self._all_words[word].add_message(message)

    def has_feature(self, feature):
        return feature in self._features

    def set_feature_value(self, original_str, feature, value):
        """
        Sets the value of a feature for a particular word
        :param feature: one of the features defined during init (e.g. "pos_tag")
        :type feature: str
        :param value: value of the feature (e.g. "noun")
        :type value: str
        """
        assert original_str in self._all_words, "Word '{}' not found in lexicon".format(original_str)
        assert feature in self._features, "Invalid feature '{}'".format(feature)

        self._all_words[original_str].set_feature_value(feature, value)

        if value not in self._features[feature]:
            self._features[feature][value] = []
        self._features[feature][value].append(original_str)

    def get_words_by_feature_value(self, feature, value):
        """
        Retrieves a list of possible words (in their original form) whose feature matches the provided value
        >>> lexicon.get_words_by_feature_value("pos_tag", "noun")
        ["apple", "orange", "cat", "dog"]

        :param feature: one of the features defined during init (e.g. "pos_tag")
        :type feature: str
        :param value: value of the feature (e.g. "noun")
        :type value: str
        :rtype: str
        """
        assert feature in self._features, "Invalid feature '{}'".format(feature)

        if value not in self._features[feature]:
            return []

        return self._features[feature][value]

    def get_feature_value_by_word(self, original_str, feature):
        """
        Retrieves the value of a feature for a particular word
        :param original_str: original form of the word
        :type original_str: str
        :param feature: one of the features defined during init (e.g. "pos_tag")
        :type feature: str
        :return: value of the feature
        :rtype: str
        """
        assert original_str in self._all_words, "Word '{}' not found in lexicon".format(original_str)
        assert feature in self._features, "Invalid feature '{}'".format(feature)

        return self._all_words[original_str].get_feature_value(feature)

    def get_messages_by_word(self, original_str):
        """
        Gets a list of all messages in which the provided word appears
        """
        assert original_str in self._all_words, "Word '{}' not found in lexicon".format(original_str)

        return self._all_words[original_str].get_messages()

    def add_message_to_word(self, original_str, message):
        assert original_str in message, "Word '{}' not found in message '{}'".format(original_str, message)
        assert original_str in self._all_words, "Word '{}' not found in lexicon".format(original_str)

        self._all_words[original_str].add_message(message)

    def add_word(self, original_str):
        """
        Adds a word not in the list of messages provided at init.
        """
        assert original_str not in self._all_words, "Word '{}' already exists in lexicon".format(original_str)

        self._all_words[original_str] = self._Word(original_str, self._features)

    def get_features(self):
        return self._features.keys()

    def get_words(self):
        return self._all_words.keys()
