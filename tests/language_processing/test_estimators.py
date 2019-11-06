import unittest
from collections import Counter

from core_ml_modules.language_processing import estimators, Lexicon, model_utils

MESSAGES = [
    ["I", "eat", "apples", "."],
    ["You", "eat", "bananas", "?"]
]
FEATURES = ["pos_tag", "canonical_form"]

class TestTokeniser(unittest.TestCase):
    def test_tokenise(self):
        tokeniser = estimators.Tokeniser()
        self.assertTrue(
            tokeniser.tokenise("I eat apples.") == ["I", "eat", "apples", "."]
        )
        self.assertTrue(
            tokeniser.tokenise("abc!@#$def    ghi") == ["abc", "!@#$", "def", "ghi"]
        )

    def test_fit_transform(self):
        tokeniser = estimators.Tokeniser()
        self.assertTrue(
            tokeniser.transform(["I eat apples.", "abc!@#$def    ghi"]) == [["I", "eat", "apples", "."], ["abc", "!@#$", "def", "ghi"]]
        )
        self.assertTrue(
            tokeniser.fit_transform(["I eat apples.", "abc!@#$def    ghi"]) == tokeniser.transform(["I eat apples.", "abc!@#$def    ghi"])
        )
        self.assertTrue(
            tokeniser.fit(["I eat apples.", "abc!@#$def    ghi"]) == tokeniser
        )

class TestNGramFrequencyExtractor(unittest.TestCase):
    def test_retrieve_lexical_form(self):
        lexicon = Lexicon(MESSAGES, FEATURES)
        lexicon.set_feature_value("I", "pos_tag", "Noun")

        regular_extractor = estimators.NGramFrequencyExtractor(lexicon)
        self.assertTrue(
            regular_extractor.retrieve_lexical_form(["I", "ate", "793", "oranges", "."]) == ["I", "ate", "793", "oranges", "."]
        )

        pos_extractor = estimators.NGramFrequencyExtractor(lexicon, form="pos_tag",
                                                           default_form=model_utils.default_form_pos)
        self.assertTrue(
            pos_extractor.retrieve_lexical_form(["I", "ate", "793", "oranges", "."]) == ["Noun", "Unk", "Num3", "Unk", "Punct"]
        )

    def test_extract_frequency_dicts(self):
        lexicon = Lexicon(MESSAGES, FEATURES)
        extractor = estimators.NGramFrequencyExtractor(lexicon)

        self.assertTrue(
            extractor.extract_frequency_dicts([["I", "eat", "oranges", "."]]) == [Counter({
                "I": 0.25, "eat": 0.25, "oranges": 0.25, ".": 0.25
            })]
        )
        self.assertTrue(
            extractor.extract_frequency_dicts([["I", "eat", "many", "many", "oranges", "."]]) == [{
                "I": 1/6, "eat": 1/6, "many": 1/3, "oranges": 1/6, ".": 1/6
            }]
        )

    def test_fit_transform(self):
        lexicon = Lexicon(MESSAGES, FEATURES)
        extractor1 = estimators.NGramFrequencyExtractor(lexicon)
        extractor2 = estimators.NGramFrequencyExtractor(lexicon)

        with self.assertRaises(AttributeError):
            extractor1.get_feature_names()

        extractor1.fit(MESSAGES, None)
        self.assertTrue(
            set(extractor1.get_feature_names()) == set(lexicon.get_words())
        )
        self.assertTrue(
            (extractor1.transform(MESSAGES).toarray() == extractor2.fit_transform(MESSAGES).toarray()).all()
        )
        self.assertTrue(
            (extractor1.transform([["I", "eat", "oranges", "."]]).toarray() == extractor2.transform([["I", "eat", "oranges", "."]]).toarray()).all()
        )

        feature_names = extractor1.get_feature_names()
        feature_map = {
            "I": 0.25,
            "eat": 0.25,
            "oranges": 0.25,
            ".": 0.25,
            "apples": 0.0,
            "You": 0.0,
            "bananas": 0.0,
            "?": 0.0
        }

        vector = []
        for feature in feature_names:
            vector.append(feature_map[feature])

        self.assertTrue(
            extractor1.transform([["I", "eat", "oranges", "."]]).toarray().tolist() == [vector]
        )

