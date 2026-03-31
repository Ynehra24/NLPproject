import unittest
from advanced_metrics import (
    damerau_levenshtein_boundary,
    count_boundary_edits,
    is_word_boundary,
    RegisterClassifier,
    apply_zone_detector
)

class TestMetrics(unittest.TestCase):
    def test_is_word_boundary(self):
        text = "Hello world"
        self.assertTrue(is_word_boundary(text, 0)) # H
        self.assertFalse(is_word_boundary(text, 1)) # e
        self.assertTrue(is_word_boundary(text, 4)) # o
        self.assertFalse(is_word_boundary(text, 5)) # ' '
        self.assertTrue(is_word_boundary(text, 6)) # w
        self.assertTrue(is_word_boundary(text, 10)) # d

    def test_count_boundary_edits(self):
        original = "cat dog"
        attacked1 = "bat dog" # c -> b (boundary)
        self.assertEqual(count_boundary_edits(original, attacked1), 1)
        
        original_mid = "cart dog"
        attacked_mid = "cast dog" # r -> s (not boundary)
        self.assertEqual(count_boundary_edits(original_mid, attacked_mid), 0)
        
        attacked_ins = "cats dog" # inserted 's' becomes boundary
        self.assertEqual(count_boundary_edits(original, attacked_ins), 1)

    def test_damerau_levenshtein_penalty_amount(self):
        # Base DL: 1. Boundary: +0.5 = 1.5
        score = damerau_levenshtein_boundary("cat", "bat", boundary_multiplier=1.5)
        self.assertEqual(score, 1.5)

        # Base DL: 1. Boundary: 0 = 1.0
        score = damerau_levenshtein_boundary("cart", "cast", boundary_multiplier=1.5)
        self.assertEqual(score, 1.0)

        # Base DL: 2. Boundary: +0.5 * 2 = 1.0. Total = 3.0
        score = damerau_levenshtein_boundary("cat dog", "bat fog", boundary_multiplier=1.5)
        self.assertEqual(score, 3.0)

    def test_zone_detector(self):
        text = "Check this https://google.com out! 🔥"
        masked = apply_zone_detector(text)
        self.assertIn("[URL]", masked)
        self.assertNotIn("https://google.com", masked)
        self.assertIn("[EMOJI]", masked)
        self.assertNotIn("🔥", masked)

    def test_register_classifier_emoji_gate(self):
        clf = RegisterClassifier()
        # Should be gated internally
        result = clf.predict("This is highly formal language 😂")
        self.assertTrue(result["gated_by_emoji"])
        self.assertEqual(result["prediction"], "Informal")

if __name__ == "__main__":
    unittest.main()
