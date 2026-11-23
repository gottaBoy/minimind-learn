"""Check that the learning script observes real training behavior."""

from contextlib import redirect_stdout
import io
import unittest

from scripts.debug_train_step import main


class DebugLessonTests(unittest.TestCase):
    def run_lesson(self, *args):
        with redirect_stdout(io.StringIO()):
            return main(["--token_rows", "0", *args])

    def test_data_only_stops_before_forward(self):
        metrics = self.run_lesson("--stop_after", "data")
        self.assertEqual(metrics["input_shape"], (1, 63))
        self.assertGreater(metrics["supervised_tokens"], 0)
        self.assertNotIn("logits_shape", metrics)

    def test_sft_mask_excludes_prompt(self):
        metrics = self.run_lesson("--task", "sft", "--max_seq_len", "128", "--stop_after", "data")
        self.assertGreater(metrics["supervised_tokens"], 0)
        self.assertLess(metrics["supervised_tokens"], metrics["nonpadding_targets"])

    def test_forward_stops_before_backward(self):
        metrics = self.run_lesson("--stop_after", "forward")
        self.assertEqual(metrics["logits_shape"], (1, 63, 6400))
        self.assertNotIn("gradient_norm", metrics)

    def test_gradients_precede_parameter_update(self):
        metrics = self.run_lesson()
        self.assertGreater(metrics["gradient_norm"], 0)
        self.assertEqual(metrics["delta_before_step"], 0)
        self.assertGreater(metrics["weight_delta"], 0)
        self.assertTrue(metrics["gradient_cleared"])
        self.assertEqual(metrics["shapes"]["layer0.k_proj"][1], (1, 63, 32))


if __name__ == "__main__":
    unittest.main()
