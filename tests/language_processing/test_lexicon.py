import unittest

from core_ml_modules.language_processing import Lexicon

MESSAGES = [
    ["I", "eat", "apples", "."],
    ["You", "eat", "bananas", "?"]
]
FEATURES = ["pos_tag", "canonical_form"]

class TestLexicon(unittest.TestCase):

    def test_init(self):
        lexicon = Lexicon(MESSAGES, FEATURES)
        self.assertTrue(lexicon.has_feature("pos_tag"))
        self.assertTrue(lexicon.has_feature("canonical_form"))
        self.assertTrue("I" in lexicon)
        self.assertTrue("You" in lexicon)
        self.assertTrue("eat" in lexicon)
        self.assertTrue("apples" in lexicon)
        self.assertTrue("bananas" in lexicon)
        self.assertTrue("." in lexicon)
        self.assertTrue("?" in lexicon)

    def test_get_set(self):
        lexicon = Lexicon(MESSAGES, FEATURES)
        self.assertTrue(
            set(lexicon.get_words()) == {"I", "eat", "apples", ".", "You", "bananas", "?"}
        )
        self.assertTrue(
            set(lexicon.get_features()) == set(FEATURES)
        )
        self.assertTrue(
            lexicon.get_words_by_feature_value("pos_tag", "noun") == []
        )
        self.assertTrue(
            lexicon.get_feature_value_by_word("I", "pos_tag") == None
        )
        self.assertTrue(
            lexicon.get_messages_by_word("eat") == MESSAGES
        )

        lexicon.set_feature_value("I", "pos_tag", "noun")
        self.assertTrue(
            lexicon.get_words_by_feature_value("pos_tag", "noun") == ["I"]
        )
        self.assertTrue(
            lexicon.get_feature_value_by_word("I", "pos_tag") == "noun"
        )

        lexicon.add_word("oranges")
        self.assertTrue(
            set(lexicon.get_words()) == {"I", "eat", "apples", ".", "You", "bananas", "?", "oranges"}
        )
        self.assertTrue(
            lexicon.get_feature_value_by_word("oranges", "pos_tag") == None
        )
        self.assertTrue(
            lexicon.get_messages_by_word("oranges") == []
        )

        lexicon.add_message_to_word("oranges", ["They", "ate", "oranges", "!"])
        self.assertTrue(
            lexicon.get_messages_by_word("oranges") == [["They", "ate", "oranges", "!"]]
        )
        with self.assertRaises(AssertionError):
            lexicon.get_messages_by_word("They")    # Words in separately added messages are not automatically added to lexicon
