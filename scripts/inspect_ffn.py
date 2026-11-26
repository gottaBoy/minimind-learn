"""Inspect the first FFN using the current 128-wide MiniMind teaching model."""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_utils import setup_seed


def matrix_shape(tensor):
    """Display batch/sequence dimensions without calling a tensor a matrix."""
    return " x ".join(str(size) for size in tensor.shape)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breakpoints", action="store_true",
                        help="Pause after forward, backward and update")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    setup_seed(42)

    # Match debug_train_step.py: fetch the batch before initializing the model.
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    dataset = PretrainDataset(ROOT / "dataset/demo_pretrain.jsonl", tokenizer, max_length=64)
    loader = DataLoader(Subset(dataset, [0]), batch_size=1, num_workers=0)
    X, Y, loss_mask = next(iter(loader))
    config = MiniMindConfig(
        vocab_size=len(tokenizer), hidden_size=128, intermediate_size=384,
        num_hidden_layers=2, max_position_embeddings=64,
    )
    model = MiniMindForCausalLM(config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ffn = model.model.layers[0].mlp
    captured = {}

    def remember(name, tensor):
        # These are actual activations from model(X), not invented FFN inputs.
        # retain_grad lets the lesson inspect gradients on non-leaf tensors.
        captured[name] = tensor
        tensor.retain_grad()

    hooks = [
        ffn.register_forward_pre_hook(lambda module, inputs: remember("x", inputs[0])),
        ffn.gate_proj.register_forward_hook(
            lambda module, inputs, output: remember("gate_raw", output)),
        ffn.act_fn.register_forward_hook(
            lambda module, inputs, output: remember("gated", output)),
        ffn.up_proj.register_forward_hook(
            lambda module, inputs, output: remember("up", output)),
        ffn.down_proj.register_forward_pre_hook(
            lambda module, inputs: remember("mixed", inputs[0])),
        ffn.down_proj.register_forward_hook(
            lambda module, inputs, output: remember("output", output)),
    ]
    try:
        result = model(X)
    finally:
        for hook in hooks:
            hook.remove()

    x, gate_raw, gated, up, mixed, output = (
        captured[name] for name in ("x", "gate_raw", "gated", "up", "mixed", "output")
    )
    rows = x.size(0) * x.size(1)
    flat_x = x.detach().reshape(rows, 128)
    W_gate = ffn.gate_proj.weight
    W_up = ffn.up_proj.weight
    W_down = ffn.down_proj.weight
    with torch.no_grad():
        torch.testing.assert_close(flat_x @ W_gate.T, gate_raw.reshape(rows, 384))
        torch.testing.assert_close(flat_x @ W_up.T, up.reshape(rows, 384))
        torch.testing.assert_close(gated * up, mixed)
        torch.testing.assert_close(mixed.reshape(rows, 384) @ W_down.T,
                                   output.reshape(rows, 128))

    per_token_loss = F.cross_entropy(
        result.logits.flatten(0, 1), Y.flatten(), reduction="none",
    ).view_as(Y)
    loss = (per_token_loss * loss_mask).sum() / loss_mask.sum()
    loss = loss + result.aux_loss
    before = {name: parameter.detach().clone() for name, parameter in ffn.named_parameters()}

    print("CPU / float32 / seed=42 / 第 0 条 demo / 第一层 FFN")
    print(f"token ID X: {matrix_shape(X)}; 有效目标: {loss_mask.sum().item()}")
    print("\n[1/3] FORWARD：实际权重和激活")
    for name, parameter in ffn.named_parameters():
        print(f"{name}: {matrix_shape(parameter)}; 参数数={parameter.numel()}")
    print("FFN 总参数:", sum(parameter.numel() for parameter in ffn.parameters()))
    print("这里的 x 是 Attention 与残差之后，经 RMSNorm 得到的浮点表示。")
    for name, tensor in captured.items():
        print(f"{name}: {matrix_shape(tensor)}")
    print(f"gate: ({rows} x 128) @ (128 x 384) = ({rows} x 384)")
    print(f"up:   ({rows} x 128) @ (128 x 384) = ({rows} x 384)")
    print(f"乘法: ({rows} x 384) * ({rows} x 384) = ({rows} x 384)，逐元素相乘")
    print(f"down: ({rows} x 384) @ (384 x 128) = ({rows} x 128)")
    print("x[0,0,:5]:", x[0, 0, :5].detach().tolist())
    print("W_gate[0,:5]:", W_gate[0, :5].detach().tolist())
    print("第一个 gate 输出:", gate_raw[0, 0, 0].item(),
          "；128 项点积:", (x[0, 0] @ W_gate[0]).item())
    for name in ("gate_raw", "gated", "up", "mixed", "output"):
        print(f"{name}[0,0,0]={captured[name][0, 0, 0].item():.6f}")
    print(f"本批实际 next-token masked loss={loss.item():.6f}")
    if args.breakpoints:
        breakpoint()

    loss.backward()
    with torch.no_grad():
        d_output = output.grad.reshape(rows, 128)
        d_gate = gate_raw.grad.reshape(rows, 384)
        d_up = up.grad.reshape(rows, 384)
        torch.testing.assert_close(W_down.grad, d_output.T @ mixed.detach().reshape(rows, 384))
        torch.testing.assert_close(W_gate.grad, d_gate.T @ flat_x)
        torch.testing.assert_close(W_up.grad, d_up.T @ flat_x)
        torch.testing.assert_close(x.grad.reshape(rows, 128), d_gate @ W_gate + d_up @ W_up)
        torch.testing.assert_close(gated.grad, mixed.grad * up)
        torch.testing.assert_close(up.grad, mixed.grad * gated)

    print("\n[2/3] BACKWARD：同一个真实 loss 沿两条支路返回")
    for name, parameter in ffn.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        print(f"{name}.grad: {matrix_shape(parameter.grad)}; "
              f"norm={parameter.grad.norm().item():.6f}; 权重尚未改变")
    print(f"dW_gate: (384 x {rows}) @ ({rows} x 128) = (384 x 128)")
    print(f"dW_up:   (384 x {rows}) @ ({rows} x 128) = (384 x 128)")
    print(f"dW_down: (128 x {rows}) @ ({rows} x 384) = (128 x 384)")
    print("矩阵乘法重算的梯度与 autograd 结果一致。")
    if args.breakpoints:
        breakpoint()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print("\n[3/3] UPDATE：AdamW 修改参数，但不改变矩阵形状")
    for name, parameter in ffn.named_parameters():
        delta = (parameter.detach() - before[name]).abs().max().item()
        assert delta > 0, name
        print(f"{name}: {matrix_shape(parameter)}; 最大变化={delta:.8f}")
    optimizer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in ffn.parameters())
    print("参数梯度已清空，更新后的权重保留；没有写入任何模型权重文件。")
    if args.breakpoints:
        breakpoint()
    return {
        "input_shape": tuple(x.shape),
        "intermediate_shape": tuple(mixed.shape),
        "output_shape": tuple(output.shape),
        "parameter_count": sum(parameter.numel() for parameter in ffn.parameters()),
        "loss": loss.item(),
    }


if __name__ == "__main__":
    main()
