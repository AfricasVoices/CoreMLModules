import unittest

from core_ml_modules.language_processing import Word

class TestWord(unittest.TestCase):

    def test_set_canonical_form(self):
        word = Word("helloo")

        self.assertTrue(word.get_original_form() == "helloo")
        self.assertTrue(word.get_canonical_form() == None)
        self.assertTrue(word.get_pos_tag() == None)

        word.set_canonical_form("hello")

        self.assertTrue(word.get_canonical_form() == "hello")
        self.assertTrue(word.get_pos_tag() == None)

        self.assertRaises(ValueError, word.set_canonical_form("hi"))

    def test_set_pos_tag(self):
        word = Word("helloo")

        self.assertTrue(word.get_original_form() == "helloo")
        self.assertTrue(word.get_canonical_form() == None)
        self.assertTrue(word.get_pos_tag() == None)

        word.set_pos_tag("INTJ")

        self.assertTrue(word.get_original_form() == "helloo")
        self.assertTrue(word.get_canonical_form() == None)
        self.assertTrue(word.get_pos_tag() == "INTJ")

        self.assertRaises(ValueError, word.set_pos_tag("NOUN"))
