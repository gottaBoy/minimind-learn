"""Regression checks for the one-day training and generation workflow."""

from contextlib import nullcontext
import importlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora
from trainer.train_utils import (
    PROJECT_ROOT, SkipBatchSampler, autocast_context, grad_scaler, project_path,
    setup_seed, torch_load,
)


def tiny_model(device="cpu", flash=True):
    config = MiniMindConfig(
        hidden_size=32, intermediate_size=64, vocab_size=64,
        num_attention_heads=4, num_key_value_heads=2,
        num_hidden_layers=2, max_position_embeddings=64, flash_attn=flash,
    )
    return MiniMindForCausalLM(config).to(device)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check_cache(self, device, flash=True, padded=False):
        setup_seed(42)
        model = tiny_model(device, flash).eval()
        tokens = torch.tensor([[3, 4, 5, 6, 7, 8]], device=device)
        mask = torch.ones_like(tokens)
        if padded:
            mask[0, 0] = 0
        with torch.no_grad():
            full = model(tokens, attention_mask=mask)
            prefill = model(tokens[:, :3], attention_mask=mask[:, :3], use_cache=True)
            # Two-token queries exercise lower-right causal masking.
            cached = model(
                tokens[:, 3:5], attention_mask=mask[:, :5],
                past_key_values=prefill.past_key_values, use_cache=True,
            )
            torch.testing.assert_close(cached.logits, full.logits[:, 3:5], atol=1e-5, rtol=1e-4)
            last = model(
                tokens[:, 5:], attention_mask=mask,
                past_key_values=cached.past_key_values, use_cache=True,
            )
            torch.testing.assert_close(last.logits, full.logits[:, 5:], atol=1e-5, rtol=1e-4)

    def test_cache_matches_full_cpu(self):
        self.check_cache("cpu")

    def test_padded_cache_matches_full_cpu(self):
        self.check_cache("cpu", padded=True)

    def test_manual_attention_matches_cache_cpu(self):
        self.check_cache("cpu", flash=False, padded=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cache_matches_full_gpu(self):
        self.check_cache("cuda")
        self.check_cache("cuda", padded=True)

    def test_generation_keeps_full_prompt(self):
        model = tiny_model().eval()
        tokens = torch.tensor([[3, 4, 5, 6]])
        lengths = []
        hook = model.model.register_forward_pre_hook(lambda module, args, kwargs: lengths.append(kwargs["input_ids"].size(1)), with_kwargs=True)
        try:
            with torch.no_grad():
                generated = model.generate(
                    tokens, max_new_tokens=3, min_new_tokens=3, do_sample=False,
                    pad_token_id=0, eos_token_id=2,
                )
        finally:
            hook.remove()
        self.assertEqual(lengths, [4, 1, 1])
        torch.testing.assert_close(generated[:, :4], tokens)

    def test_cached_and_uncached_generation_agree(self):
        model = tiny_model().eval()
        tokens = torch.tensor([[3, 4, 5, 6]])
        options = dict(max_new_tokens=4, min_new_tokens=4, do_sample=False, pad_token_id=0, eos_token_id=2)
        torch.testing.assert_close(
            model.generate(tokens, use_cache=True, **options),
            model.generate(tokens, use_cache=False, **options),
        )

    def test_forward_outputs_are_independent(self):
        model = tiny_model().eval()
        first = model(torch.tensor([[3, 4, 5]]))
        model(torch.tensor([[3]]))
        self.assertEqual(first.logits.shape, (1, 3, 64))


class TrainingTests(unittest.TestCase):
    def test_accumulation_and_final_checkpoint(self):
        for module_name in ["train_pretrain", "train_full_sft", "train_lora"]:
            with self.subTest(module=module_name), tempfile.TemporaryDirectory() as directory:
                setup_seed(42)
                module = importlib.import_module(f"trainer.{module_name}")
                model = tiny_model()
                if module_name == "train_lora":
                    apply_lora(model, rank=2)
                    for name, parameter in model.named_parameters():
                        parameter.requires_grad_("lora" in name)
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.SGD(parameters, lr=0.01)
                x = torch.tensor([[3, 4, 5], [6, 7, 8], [9, 10, 11]])
                dataset = TensorDataset(x, x + 1, torch.ones_like(x))
                loader = DataLoader(dataset, batch_size=1)
                args = SimpleNamespace(
                    device="cpu", accumulation_steps=2, epochs=1, learning_rate=0.01,
                    grad_clip=1e6, log_interval=100, save_interval=100, max_steps=0,
                    save_dir=directory, checkpoint_dir=directory,
                    save_weight="test", lora_name="test",
                )
                # The reference explicitly averages batches [1,2], then [3].
                reference = tiny_model()
                if module_name == "train_lora":
                    apply_lora(reference, rank=2)
                    for name, parameter in reference.named_parameters():
                        parameter.requires_grad_("lora" in name)
                reference.load_state_dict(model.state_dict())
                ref_optimizer = torch.optim.SGD(
                    [p for p in reference.parameters() if p.requires_grad], lr=0.01,
                )
                for group in [[0, 1], [2]]:
                    ref_optimizer.zero_grad()
                    for index in group:
                        loss = torch.nn.functional.cross_entropy(
                            reference(x[index:index + 1]).logits.flatten(0, 1),
                            (x[index:index + 1] + 1).flatten(),
                        )
                        (loss / len(group)).backward()
                    lr = module.get_lr(group[-1] + 1, 3, 0.01)
                    ref_optimizer.param_groups[0]["lr"] = lr
                    ref_optimizer.step()
                with patch.multiple(
                    module, args=args, model=model, lm_config=model.config,
                    optimizer=optimizer, scaler=grad_scaler("cpu", torch.float32),
                    autocast_ctx=nullcontext(), create=True,
                ):
                    extra = [parameters] if module_name == "train_lora" else []
                    module.train_epoch(0, loader, 3, *extra)
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                resume = torch_load(Path(directory) / "test_32_resume.pth")
                self.assertEqual(resume["step"], 3)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, resume["model"][key])

    def test_single_batch_is_saved_and_limit_is_obeyed(self):
        module = importlib.import_module("trainer.train_pretrain")
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model()
            x = torch.tensor([[3, 4, 5], [6, 7, 8]])
            loader = DataLoader(TensorDataset(x, x + 1, torch.ones_like(x)), batch_size=1)
            args = SimpleNamespace(
                device="cpu", accumulation_steps=8, epochs=1, learning_rate=0.01,
                grad_clip=1.0, log_interval=100, save_interval=100, max_steps=1,
                save_dir=directory, checkpoint_dir=directory, save_weight="limit",
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            with patch.multiple(
                module, args=args, model=model, lm_config=model.config,
                optimizer=optimizer, scaler=grad_scaler("cpu", torch.float32),
                autocast_ctx=nullcontext(), create=True,
            ):
                module.train_epoch(0, loader, 2)
            self.assertEqual(torch_load(Path(directory) / "limit_32_resume.pth")["step"], 1)
            self.assertTrue(optimizer.state)

    def test_resume_skips_exact_batch_count(self):
        sampler = SkipBatchSampler(range(7), batch_size=2, skip_batches=2)
        self.assertEqual(list(sampler), [[4, 5], [6]])
        self.assertEqual(len(sampler), 2)

    def test_project_paths_do_not_depend_on_cwd(self):
        self.assertEqual(project_path("out"), str(PROJECT_ROOT / "out"))
        self.assertEqual(project_path("/tmp/weights"), "/tmp/weights")
        self.assertEqual(Path(project_path("../external")).resolve(), PROJECT_ROOT.parent / "external")
        with autocast_context("cpu", torch.bfloat16):
            self.assertEqual((torch.ones(1) + 1).item(), 2)

    def test_demo_data_has_training_tokens(self):
        from dataset.lm_dataset import PretrainDataset, SFTDataset
        tokenizer = AutoTokenizer.from_pretrained(PROJECT_ROOT / "model")
        for dataset in [
            PretrainDataset(PROJECT_ROOT / "dataset/demo_pretrain.jsonl", tokenizer, 128),
            SFTDataset(PROJECT_ROOT / "dataset/demo_sft.jsonl", tokenizer, 128),
        ]:
            for index in range(len(dataset)):
                self.assertGreater(dataset[index][2].sum().item(), 0)


class DataPreparationTests(unittest.TestCase):
    def test_subset_stops_at_limit(self):
        from scripts.prepare_data import write_subset
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "subset.jsonl"
            stream = io.StringIO('{"text":"first"}\n{"text":"second"}\ninvalid trailing data\n')
            self.assertEqual(write_subset(stream, output, "pretrain", 2), 2)
            self.assertEqual(len(output.read_text().splitlines()), 2)

    def test_failed_subset_preserves_existing_file(self):
        from scripts.prepare_data import write_subset
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "subset.jsonl"
            output.write_text("existing")
            with self.assertRaises(ValueError):
                write_subset(io.StringIO('{"wrong": 1}\n'), output, "pretrain", 2)
            self.assertEqual(output.read_text(), "existing")
            self.assertFalse(output.with_name("subset.jsonl.part").exists())


if __name__ == "__main__":
    unittest.main()
