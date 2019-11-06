import unittest
import numpy as np

from core_ml_modules.language_processing import model_utils, Lexicon

class TestModelUtils(unittest.TestCase):
    def test_select_predicted_scores(self):
        binary_scores = np.array([0.1, 0.2, 0.3, 0.4])
        multiclass_scores = np.array([
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.3, 0.0],
            [0.4, 0.0, 0.0, 0.0]
        ])

        predicted_labels = ["two", "four", "three", "one"]
        possible_labels = ["one", "two", "three", "four"]

        self.assertTrue(
            model_utils.select_predicted_scores(binary_scores, predicted_labels, possible_labels).all() ==
            model_utils.select_predicted_scores(multiclass_scores, predicted_labels, possible_labels).all()
        )

    def test_default_form_pos(self):
        lexicon = Lexicon([[]], [])
        self.assertTrue(
            model_utils.default_form_pos("12345", lexicon) == "Num5"
        )
        self.assertTrue(
            model_utils.default_form_pos(".@$!@#$(}{", lexicon) == "Punct"
        )
        self.assertTrue(
            model_utils.default_form_pos("abc", lexicon) == "Unk"
        )
        self.assertTrue(
            model_utils.default_form_pos("abc1234", lexicon) == "Unk"
        )


