"""Inspect one real MiniMind batch, forward pass, backward pass, and update.

This script is deliberately small: it follows one sample through the same
sequence used by training, then stops at four observable stages:
DATA -> FORWARD -> BACKWARD -> UPDATE.
"""

import argparse
from functools import partial
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

# `__file__` is this script's path:
#   <project_root>/scripts/debug_train_step.py
# Resolve symlinks and use `parents[1]` to reach `<project_root>`, so the
# script works regardless of the directory from which it is launched.
ROOT = Path(__file__).resolve().parents[1]

# The teaching script is also runnable as a standalone file. Put the project
# root first on Python's import path so `dataset`, `model`, and `trainer`
# resolve to this repository rather than to an unrelated installed package.
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset, SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_utils import setup_seed


def record_shape(module, inputs, output, *, name, shapes):
    """Record a module's input/output shapes for the forward-pass lesson.

    Forward hooks receive the module, its positional inputs, and its output.
    Some MiniMind modules return tuples, so the first tensor is used when
    necessary.
    """
    tensor = output[0] if isinstance(output, tuple) else output
    shapes[name] = (tuple(inputs[0].shape), tuple(tensor.shape))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # These options let the lesson stop after data preparation, forward,
    # backward, or the optimizer update without changing the script.
    parser.add_argument("--task", choices=["pretrain", "sft"], default="pretrain",
                        help="Use next-token pretraining data or SFT conversation data")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Run the small teaching model on CPU or CUDA")
    parser.add_argument("--max_seq_len", type=int, default=64,
                        help="Length after padding/truncation; X and Y have max_seq_len - 1 positions")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Which demo sample to inspect")
    parser.add_argument("--token_rows", type=int, default=16,
                        help="How many X/Y rows to print; 0 prints the shapes only")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate used only for the in-memory update")
    parser.add_argument("--stop_after", choices=["data", "forward", "backward", "step"], default="step",
                        help="Return after the selected teaching stage")
    parser.add_argument("--breakpoints", action="store_true",
                        help="Pause at each stage for pdb or the VS Code debugger")
    args = parser.parse_args(argv)
    if args.max_seq_len < 2 or args.token_rows < 0:
        parser.error("max_seq_len must be >= 2 and token_rows must be >= 0")
    if args.learning_rate <= 0:
        parser.error("learning_rate must be greater than zero")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    # Keep the CPU lesson quick and reproducible. `42` is an arbitrary label,
    # not a special ML value; setup_seed applies it before model construction
    # so the initial weights and later random operations start the same.
    # It seeds Python, NumPy, PyTorch, and CUDA generators, and asks cuDNN for
    # deterministic algorithms when CUDA is available.
    if args.device == "cpu":
        torch.set_num_threads(2)
    setup_seed(42)

    # The tokenizer maps text <-> integer token IDs. The dataset then creates
    # the shifted X/Y pair and a mask saying which target positions count.
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    dataset_type = PretrainDataset if args.task == "pretrain" else SFTDataset
    data_path = ROOT / "dataset" / f"demo_{args.task}.jsonl"
    dataset = dataset_type(data_path, tokenizer, max_length=args.max_seq_len)
    if not 0 <= args.sample_index < len(dataset):
        parser.error(f"sample_index must be between 0 and {len(dataset) - 1}")

    sample = dataset.samples[args.sample_index]
    loader = DataLoader(Subset(dataset, [args.sample_index]), batch_size=1, num_workers=0)
    # X is the context row and Y is the same token sequence shifted left:
    # X[0, t] is used to predict Y[0, t]. The loader adds the batch dimension.
    X, Y, loss_mask = (tensor.to(args.device) for tensor in next(iter(loader)))
    supervised_tokens = int(loss_mask.sum().item())
    if supervised_tokens == 0:
        raise ValueError("This sample has no supervised tokens; increase --max_seq_len")
    # This invariant catches a broken causal-language-model data pipeline
    # before any model computation starts.
    torch.testing.assert_close(X[:, 1:], Y[:, :-1])
    metrics = {
        "task": args.task,
        "device": args.device,
        "input_shape": tuple(X.shape),
        "supervised_tokens": supervised_tokens,
        "nonpadding_targets": int((Y != tokenizer.pad_token_id).sum().item()),
    }

    print(f"task={args.task}; device={args.device}; dtype=float32; seed=42")
    print("\n[1/4] DATA: raw sample -> tokenizer -> X, Y, loss_mask")
    print(sample)
    print(f"X={tuple(X.shape)}; Y={tuple(Y.shape)}; loss_mask={tuple(loss_mask.shape)}")
    print(f"supervised_tokens={supervised_tokens}; X[:, 1:] == Y[:, :-1]: True")
    print("index | X id | X token -> Y id | Y token | loss mask")
    for index in range(min(args.token_rows, X.size(1))):
        x_id, y_id = X[0, index].item(), Y[0, index].item()
        x_token = tokenizer.decode([x_id])
        y_token = tokenizer.decode([y_id])
        print(f"{index:5} | {x_id:4} | {x_token!r} -> {y_id:4} | {y_token!r} | {loss_mask[0, index].item()}")
    if args.breakpoints:
        breakpoint()
    if args.stop_after == "data":
        return metrics

    # Use a two-layer, 128-wide model so the lesson is fast and easy to
    # inspect. It has the same vocabulary and tensor contracts as training,
    # but it is not the full 25.83M-parameter training model.
    config = MiniMindConfig(
        vocab_size=len(tokenizer), hidden_size=128, intermediate_size=384,
        num_hidden_layers=2, max_position_embeddings=args.max_seq_len,
    )
    model = MiniMindForCausalLM(config).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    shapes = {}
    # Hooks observe the tensor boundary of each module without changing its
    # output. The table printed below makes [B, T, H] -> [B, T, V] visible.
    observed_modules = {
        "embedding": model.model.embed_tokens,
        "layer0.q_proj": model.model.layers[0].self_attn.q_proj,
        "layer0.k_proj": model.model.layers[0].self_attn.k_proj,
        "layer0.v_proj": model.model.layers[0].self_attn.v_proj,
        "layer0.attention": model.model.layers[0].self_attn,
        "layer0.ffn": model.model.layers[0].mlp,
        "layer0": model.model.layers[0],
        "lm_head": model.lm_head,
    }
    hooks = [module.register_forward_hook(partial(record_shape, name=name, shapes=shapes))
             for name, module in observed_modules.items()]
    try:
        result = model(X)
    finally:
        # Hooks are only for this observation. Remove them before any later
        # computation so they cannot affect subsequent forwards.
        for hook in hooks:
            hook.remove()

    # logits has shape [B, T, V]. At each position it contains one raw score
    # for every vocabulary token; it is not a probability distribution yet.
    logits = result.logits
    # Cross-entropy accepts raw logits. Flatten B and T so every token
    # prediction is scored, then reshape to recover one loss per position.
    per_token_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), Y.flatten(), reduction="none",
    ).view_as(Y)
    # Padding (and, for SFT, prompt positions) have mask 0 and do not affect
    # the average. aux_loss is zero for this dense teaching model but keeps
    # the code compatible with the model's optional MoE output.
    loss = (per_token_loss * loss_mask).sum() / loss_mask.sum()
    loss = loss + result.aux_loss
    if not torch.isfinite(loss):
        raise ValueError("The loss is not finite")
    metrics.update(logits_shape=tuple(logits.shape), loss=loss.item(), shapes=shapes)

    print("\n[2/4] FORWARD: X -> embedding -> transformer -> logits -> masked loss")
    for name, (input_shape, output_shape) in shapes.items():
        print(f"{name}: {input_shape} -> {output_shape}")
    print(f"per_token_loss={tuple(per_token_loss.shape)}; masked_loss={loss.item():.6f}")
    print(f"weight.grad is None before backward: {model.lm_head.weight.grad is None}")
    if args.breakpoints:
        breakpoint()
    if args.stop_after == "forward":
        return metrics

    # Save one parameter tensor before backward so we can prove the order:
    # backward creates gradients, while optimizer.step changes the weights.
    weight_before = model.lm_head.weight.detach().clone()
    loss.backward()
    gradient = model.lm_head.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise ValueError("Expected a finite lm_head gradient")
    gradient_norm = gradient.norm().item()
    delta_before_step = (model.lm_head.weight.detach() - weight_before).abs().max().item()
    metrics.update(gradient_norm=gradient_norm, delta_before_step=delta_before_step)

    print("\n[3/4] BACKWARD: loss.backward() creates gradients")
    print(f"lm_head.weight.grad.norm()={gradient_norm:.6f}")
    print(f"weight_delta_before_step={delta_before_step:.8f}")
    if args.breakpoints:
        breakpoint()
    if args.stop_after == "backward":
        return metrics

    # Clip unusually large gradients, apply the update, then clear gradients
    # so the next batch will not accumulate this batch by accident.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    weight_delta = (model.lm_head.weight.detach() - weight_before).abs().max().item()
    if weight_delta == 0:
        raise ValueError("Expected the optimizer to update lm_head weights")
    optimizer.zero_grad(set_to_none=True)
    metrics.update(weight_delta=weight_delta, gradient_cleared=model.lm_head.weight.grad is None)

    print("\n[4/4] UPDATE: clip gradients -> optimizer.step() -> zero_grad()")
    print(f"weight_delta_after_step={weight_delta:.8f}")
    print(f"weight.grad is None after zero_grad: {model.lm_head.weight.grad is None}")
    print("Completed one update in memory. No model weights were written.")
    if args.breakpoints:
        breakpoint()
    return metrics


if __name__ == "__main__":
    main()
