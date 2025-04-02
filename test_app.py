import os
import unittest
from PIL import Image
import io

# Import functions from your modules
from load import load_model
from app import load_flower_info, classify_image, log_feedback

class TestLoadModel(unittest.TestCase):
    def test_load_model(self):
        # Check if the model file exists; if not, skip this test.
        model_path = "model.pth"
        if not os.path.exists(model_path):
            self.skipTest(f"{model_path} not found.")
        model = load_model(model_path)
        # Verify the model has the expected fc layer with 102 outputs.
        self.assertTrue(hasattr(model, "fc"))
        self.assertEqual(model.fc.out_features, 102)

class TestLoadFlowerInfo(unittest.TestCase):
    def test_load_flower_info(self):
        # Check if flower.json exists; if not, skip this test.
        if not os.path.exists("flower.json"):
            self.skipTest("flower.json not found.")
        flower_info = load_flower_info("flower.json")
        self.assertIsInstance(flower_info, dict)
        # Check that at least one flower is loaded.
        self.assertGreaterEqual(len(flower_info), 1)
        # Optionally, check that one of the expected keys is in one entry.
        sample_key = next(iter(flower_info))
        self.assertIn("name", flower_info[sample_key])
        self.assertIn("scientific_name", flower_info[sample_key])

class TestClassifyImage(unittest.TestCase):
    def test_classify_image_output(self):
        # Create a dummy white image with PIL
        dummy_image =Image.open("images.jpeg")
        try:
            predicted_class, info = classify_image(dummy_image)
        except Exception as e:
            self.fail(f"classify_image raised an exception: {e}")
        # Check that predicted_class is an integer.
        self.assertIsInstance(predicted_class, int)
        # info might be None if predicted class is not in flower_info,
        # but if it's not None, check for expected keys.
        if info is not None:
            for key in ["name", "scientific_name", "genus", "fun_fact", "where_found"]:
                self.assertIn(key, info)

class TestLogFeedback(unittest.TestCase):
    def test_log_feedback(self):
        # Use a temporary file for testing log_feedback
        test_log_file = "test_feedback.log"
        # Ensure the file doesn't exist before the test
        if os.path.exists(test_log_file):
            os.remove(test_log_file)
        
        # Monkey-patch the built-in open within log_feedback's scope
        # We'll temporarily replace it so it writes to our test file.
        import builtins
        original_open = builtins.open
        try:
            def dummy_open(filename, mode="r", *args, **kwargs):
                # Redirect all file writes to our test log file.
                return original_open(test_log_file, mode, *args, **kwargs)
            builtins.open = dummy_open

            # Call log_feedback with test data
            log_feedback(5, "Test Flower")
        finally:
            builtins.open = original_open

        # Verify the test log file now contains the expected text.
        with open(test_log_file, "r") as f:
            contents = f.read()
        os.remove(test_log_file)
        self.assertIn("Predicted: 5, Correction: Test Flower", contents)

if __name__ == "__main__":
    unittest.main()
