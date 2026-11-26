"""Guard the practical course paths and checkpoint-resume exercise."""

from contextlib import chdir, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from eval_llm import init_model
from model.model_lora import apply_lora, save_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from scripts.check_course import check_documents
from scripts.check_doc_examples import examples, main as check_doc_examples
from scripts.lesson_lab import lesson_resume
from trainer.train_utils import PROJECT_ROOT


class DocumentationCheckTests(unittest.TestCase):
    def test_inline_logs_and_code_are_not_checked_as_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "readme.md").write_text(
                "`Epoch:[1/1](1/3)`\n"
                "```text\n[example](missing)\n```\n"
                "[data](data.txt)\n", encoding="utf-8",
            )
            (root / "data.txt").touch()
            errors, counts = check_documents(root)
            self.assertEqual(errors, [])
            self.assertEqual(counts["links"], 1)

    def test_broken_link_invalid_python_and_unclosed_fence_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "readme.md").write_text(
                "[missing](absent.md)\n"
                "```python\nif True:\n```\n"
                "```text\nunfinished\n", encoding="utf-8",
            )
            errors, _ = check_documents(root)
            self.assertEqual(len(errors), 3)
            self.assertTrue(any("missing local link" in error for error in errors))
            self.assertTrue(any("Python syntax" in error for error in errors))
            self.assertTrue(any("unclosed code fence" in error for error in errors))

    def test_example_runner_selects_only_marked_python_and_preserves_context(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "examples.md"
            document.write_text(
                "```python\nraise RuntimeError('debugger fragment')\n```\n"
                "```bash\n# doc-example: shell\nexit 1\n```\n"
                "```python\n# doc-example: setup\nvalue = 40\n```\n"
                "```python\n# doc-example: use\nassert value + 2 == 42\n```\n",
                encoding="utf-8",
            )
            self.assertEqual(len(examples(document)), 2)
            with redirect_stdout(io.StringIO()) as output:
                result = check_doc_examples(["--document", str(document)])
            self.assertEqual(result, 0)
            self.assertIn("2 executable blocks, 0 failures", output.getvalue())

    def test_example_runner_reports_failure_with_document_location(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "broken.md"
            document.write_text(
                "```python\n# doc-example: failure\nassert False, 'expected failure'\n```\n",
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()) as output:
                result = check_doc_examples(["--document", str(document)])
            self.assertEqual(result, 1)
            self.assertIn("expected failure", output.getvalue())
            self.assertIn('line 3', output.getvalue())

    def test_example_runner_rejects_unclosed_fences(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "unfinished.md"
            document.write_text("```python\n# doc-example: unfinished\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unclosed code fence"):
                examples(document)


class AdapterInferenceTests(unittest.TestCase):
    def test_adapter_can_be_loaded_from_an_independent_directory(self):
        torch.set_num_threads(2)
        config = MiniMindConfig(hidden_size=32, num_hidden_layers=2)
        model = MiniMindForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(model.state_dict(), root / "full_sft_32.pth")
            apply_lora(model)
            with torch.no_grad():
                model.model.layers[0].self_attn.q_proj.lora.B.weight.normal_(std=0.02)
            adapter = root / "attachments/custom.pth"
            adapter.parent.mkdir()
            save_lora(model, adapter)
            tokens = torch.tensor([[47, 261, 75]])
            with torch.no_grad():
                expected = model(tokens).logits
            for location in (str(adapter), os.path.relpath(adapter, PROJECT_ROOT)):
                args = SimpleNamespace(
                    load_from="model", save_dir=str(root), weight="full_sft",
                    hidden_size=32, num_hidden_layers=2, use_moe=0,
                    inference_rope_scaling=False, lora_weight="None", lora_path=location,
                    device="cpu",
                )
                # Relative adapter paths must work even outside the repository.
                with self.subTest(path=location), chdir(root), redirect_stdout(io.StringIO()):
                    restored, _ = init_model(args)
                with torch.no_grad():
                    torch.testing.assert_close(restored(tokens).logits, expected, rtol=0, atol=0)

    def test_legacy_adapter_name_still_resolves_from_project_root(self):
        args = SimpleNamespace(
            load_from="model", save_dir="out/course_test", weight="full_sft",
            hidden_size=32, num_hidden_layers=2, use_moe=0,
            inference_rope_scaling=False, lora_weight="lora_identity", lora_path=None,
            device="cpu",
        )
        config = MiniMindConfig(hidden_size=32, num_hidden_layers=2)
        model = MiniMindForCausalLM(config)
        with patch("eval_llm.torch_load", return_value=model.state_dict()), \
                patch("eval_llm.load_lora") as loader, redirect_stdout(io.StringIO()):
            init_model(args)
        loader.assert_called_once()
        self.assertEqual(Path(loader.call_args.args[1]),
                         PROJECT_ROOT / "out/course_test/lora/lora_identity_32.pth")


class ResumeLessonTests(unittest.TestCase):
    def test_real_checkpoint_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(2)
        with redirect_stdout(io.StringIO()) as output:
            lesson_resume()
        self.assertIn("只处理 batch 3", output.getvalue())


if __name__ == "__main__":
    unittest.main()
