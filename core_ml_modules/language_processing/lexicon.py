import math

from . import Word

class Lexicon:
    """
    Object containing lexical information on all words in the dataset

    :ivar all_words: a list of all words in the dataset as Word objects
    :type all_words: list(Word)
    :ivar original_form_dict: dictionary where keys are the original forms as strings,
                              and the values are the corresponding Word objects
    :type original_form_dict: dict(str, Word)
    :ivar original_form_dict: dictionary where keys are the canonical forms as strings,
                              and the values are lists of corresponding Word objects
    :type original_form_dict: dict(str, list(Word))
    :ivar original_form_dict: dictionary where keys are the part-of-speech tags as strings,
                              and the values are lists of corresponding Word objects
    :type original_form_dict: dict(str, list(Word))
    :ivar contexts: a list of context word windows which include the target words in the middle
    :type contexts: list(list(Word))
    :ivar context_windows_dict: dictionary where keys are the original forms of the middle word as strings,
                        and the values are lists of corresponding context Word windows
    :type context_windows_dict: dict(str, list(list(Word)))
    """

    def __init__(self, messages, extract_contexts_function):
        """
        Initialises the lexicon
        :param messages: a list of all messages
        :type messages: list(list(str))
        :param extract_contexts_function: Function that extracts contexts.
                                          Target word must be in the center of each context window
        :type extract_contexts_function: function((Lexicon, list(list(str))) => list(list(str)))
        """
        self.all_words = self._extract_words(messages)

        self.original_form_dict = {}
        self.canonical_form_dict = {}
        self.pos_tag_dict = {}
        self._sort_words_into_dicts()

        self.context_windows = extract_contexts_function(self, messages)
        self.context_windows_dict = {}
        self._sort_contexts_into_dicts()

    def _extract_words(self, messages):
        """
        Extract all words as Words in messages
        :type messages: list(list(str))
        :rtype: list(Word)
        """
        words = []
        for message in messages:
            for word in message:
                if word not in words:
                    words.append(Word(word))

        return words

    def _sort_words_into_dicts(self):
        """
        Sort word objects into dictionaries with appropriate keys
        """
        for word in self.all_words:
            original_form = word.original_form
            canonical_form = word.canonical_form
            pos_tag = word.pos_tag

            self.original_form_dict[original_form] = word

            if canonical_form:
                if canonical_form not in self.canonical_form_dict:
                    self.canonical_form_dict[canonical_form] = []
                self.canonical_form_dict[canonical_form].append(word)

            if pos_tag:
                if pos_tag not in self.pos_tag_dict:
                    self.pos_tag_dict[pos_tag] = []
                self.pos_tag_dict[pos_tag].append(word)

    def _sort_contexts_into_dicts(self):
        """
        Sort context windows into dictionaries with target words as keys
        """
        for context in self.context_windows:
            if len(context) < 3 or len(context) % 2 == 0:
                raise ValueError("Context windows must contain the target word as well as its neighbours, "
                                 "so the context window size must be an odd number greater than 1")

            target_word = context[len(context) / 2]

            if target_word not in self.context_windows_dict:
                self.context_windows_dict[target_word] = []
            self.context_windows_dict[target_word].append(context)

    def get_word_by_original_form(self, original_form):
        return self.original_form_dict[original_form]

    def get_words_by_canonical_form(self, canonical_form):
        return self.canonical_form_dict[canonical_form]

    def get_words_by_pos_tag(self, pos_tag):
        return self.pos_tag_dict[pos_tag]

    def _cluster_word(self, original_form, canonical_form):
        """
        Assigns a canonical form to a word
        :type original_form: str
        :type canonical_form: str
        """
        word_object = self.get_word_by_original_form(original_form)
        word_object.set_canonical_form(canonical_form)

        if canonical_form not in self.canonical_form_dict:
            self.canonical_form_dict[canonical_form] = [word_object]
        else:
            self.canonical_form_dict[canonical_form].append(word_object)

    def add_cluster(self, canonical_form, detect_function):
        """
        Manually add a cluster which can be easily identified, e.g. with a regular expressions or a pre-defined list.
        :param canonical_form: Head of the cluster to be formed
        :type canonical_form: str
        :param detect_function: Function to detect whether a word should be added to this cluster
        :type detect_function: function((Lexicon, str) => bool)
        """
        for word in self.all_words:
            if detect_function(self, word.original_form):
                self._cluster_word(word.original_form, canonical_form)

    def cluster(self, distance_function, threshold_function, filter_function=lambda lexicon: lexicon.all_words):
        """
        Clusters words using the given distance function
        :param distance_function: Computes distance between 2 words. Should return math.inf if exact distance has not
                                  been computed but is too large for clustering.
        :type distance_function: function((Lexicon, str, str, float) => float)
        :param threshold_function: Computes an appropriate threshold given 2 words. Word will be clustered if
                                   minimum distance is less than threshold.
                                   Should return 0 if closest_canonical_form is None
        :type threshold_function: function((Lexicon, str, str, float) => float)
        :param filter_function: Filters words that should not be clustered, such as low frequency words or ones
                                that can be clustered based on regular expressions (e.g. numbers).
        :type filter_function: function(Lexicon => list(Word))
        """
        filtered_words = filter_function(self)

        for word in filtered_words:

            minimum_distance = math.inf
            closest_canonical_form = None

            for canonical_form in self.canonical_form_dict.keys():
                distance = distance_function(self, word.original_form, canonical_form, minimum_distance)

                if distance < minimum_distance:
                    minimum_distance = distance
                    closest_canonical_form = canonical_form

            # Successfully found canonical form that is sufficiently close to target word
            if (closest_canonical_form
            and minimum_distance < threshold_function(self, word.original_form, closest_canonical_form, minimum_distance)):

                self._cluster_word(word.original_form, closest_canonical_form)

            # Cannot find appropriate canonical form, so create a new cluster
            else:
                self._cluster_word(word.original_form, word.original_form)
