"""Follow real token IDs through Embedding, Q/K/V and the vocabulary logits."""

import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

# Resolve paths from this file so the script also works outside the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import (
    MiniMindConfig, MiniMindForCausalLM, apply_rotary_pos_emb, repeat_kv,
)
from trainer.train_utils import setup_seed


def shape(tensor):
    return " x ".join(str(size) for size in tensor.shape)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breakpoints", action="store_true",
                        help="Inspect saved activations at three lesson stops")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    setup_seed(42)

    # Keep batch/model creation order identical to debug_train_step.py.
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    dataset = PretrainDataset(ROOT / "dataset/demo_pretrain.jsonl", tokenizer, max_length=64)
    loader = DataLoader(Subset(dataset, [0]), batch_size=1, num_workers=0)
    X, Y, loss_mask = next(iter(loader))
    config = MiniMindConfig(
        vocab_size=len(tokenizer), hidden_size=128, intermediate_size=384,
        num_hidden_layers=2, max_position_embeddings=64,
    )
    model = MiniMindForCausalLM(config).train()
    attn = model.model.layers[0].self_attn
    captured = {}

    def remember(name, tensor):
        # Hooks observe the actual model(X) call without replacing any layer.
        captured[name] = tensor

    hooks = [
        model.model.embed_tokens.register_forward_hook(
            lambda module, inputs, output: remember("embedding", output)),
        attn.register_forward_pre_hook(
            lambda module, inputs: remember("a", inputs[0])),
        attn.q_proj.register_forward_hook(
            lambda module, inputs, output: remember("Q", output)),
        attn.k_proj.register_forward_hook(
            lambda module, inputs, output: remember("K", output)),
        attn.v_proj.register_forward_hook(
            lambda module, inputs, output: remember("V", output)),
        attn.o_proj.register_forward_pre_hook(
            lambda module, inputs: remember("merged", inputs[0])),
        attn.o_proj.register_forward_hook(
            lambda module, inputs, output: remember("attention_output", output)),
        model.model.layers[0].register_forward_hook(
            lambda module, inputs, output: remember("block0", output[0])),
        model.model.layers[1].register_forward_hook(
            lambda module, inputs, output: remember("block1", output[0])),
        model.lm_head.register_forward_pre_hook(
            lambda module, inputs: remember("head_input", inputs[0])),
    ]
    try:
        result = model(X)
    finally:
        for hook in hooks:
            hook.remove()

    B, T = X.shape
    H = config.hidden_size
    N = B * T
    E = model.model.embed_tokens.weight
    embedding = captured["embedding"]
    token_id = X[0, 0].item()
    target_id = Y[0, 0].item()
    assert tuple(X.shape) == (1, 63)
    assert tuple(embedding.shape) == (1, 63, 128)
    assert E is model.lm_head.weight
    torch.testing.assert_close(embedding, E[X], rtol=0, atol=0)

    print("CPU / float32 / seed=42 / 第 0 条 demo / model/ 的 6400 词表")
    print("原文:", dataset.samples[0]["text"])
    print("三个停点查看同一次完整前向保存的结果；本脚本不反向、不更新权重。")
    print("\n[1/3] Embedding：整数 ID 查行")
    print(f"X: {shape(X)}; E: {shape(E)}; E[X]: {shape(embedding)}")
    print("X 前五项:", X[0, :5].tolist())
    print(f"X[0,0]={token_id}, token={tokenizer.decode([token_id])!r}")
    print(f"Y[0,0]={target_id}, token={tokenizer.decode([target_id])!r}")
    print(f"E[{token_id},:5]:", E[token_id, :5].tolist())
    print("embedding[0,0,:5]:", embedding[0, 0, :5].tolist())
    print(f"共享表参数数={E.numel()}，输入与输出只计一份")
    if args.breakpoints:
        breakpoint()

    # Attention sees the normalized embedding, rather than the integer IDs.
    a = captured["a"]
    Q, K, V = (captured[name] for name in ("Q", "K", "V"))
    Wq, Wk, Wv, Wo = (layer.weight for layer in (
        attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
    ))
    torch.testing.assert_close(a, model.model.layers[0].input_layernorm(embedding))
    for weight, projection in ((Wq, Q), (Wk, K), (Wv, V)):
        torch.testing.assert_close(
            a.reshape(N, H) @ weight.T, projection.reshape(N, -1),
        )
    assert tuple(K.shape) == tuple(V.shape) == (1, 63, 32)

    # Reconstruct attention using the production RoPE and head-repeat helpers.
    q_split = Q.view(B, T, attn.n_local_heads, attn.head_dim)
    k_split = K.view(B, T, attn.n_local_kv_heads, attn.head_dim)
    v_split = V.view(B, T, attn.n_local_kv_heads, attn.head_dim)
    q_rope, k_rope = apply_rotary_pos_emb(
        q_split, k_split, model.model.freqs_cos[:T], model.model.freqs_sin[:T],
    )
    q_heads = q_rope.transpose(1, 2)
    k_heads = repeat_kv(k_rope, attn.n_rep).transpose(1, 2)
    v_heads = repeat_kv(v_split, attn.n_rep).transpose(1, 2)
    scores = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    visible = torch.ones(T, T, dtype=torch.bool).tril()
    masked_scores = scores.masked_fill(~visible, float("-inf"))
    attention_weights = F.softmax(masked_scores.float(), dim=-1).type_as(q_heads)
    attended = attention_weights @ v_heads
    merged = attended.transpose(1, 2).contiguous().view(B, T, H)
    # SDPA may sum float32 values in a different order from explicit matmuls.
    torch.testing.assert_close(merged, captured["merged"], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        captured["merged"] @ Wo.T, captured["attention_output"],
    )
    torch.testing.assert_close(attention_weights[0, 0, 0], visible[0].float())
    torch.testing.assert_close(k_heads[:, 0], k_heads[:, 3], rtol=0, atol=0)

    print("\n[2/3] 第一层 Attention：投影、分头、汇总")
    print(f"a = RMSNorm(embedding): {shape(a)}")
    for name, weight, projection in (("Q", Wq, Q), ("K", Wk, K), ("V", Wv, V)):
        print(f"W{name}: {shape(weight)}; 参数数={weight.numel()}")
        print(f"{name}: ({N} x {H}) @ ({shape(weight.T)}) "
              f"= {N} x {weight.size(0)} -> {shape(projection)}")
    print(f"Wo: {shape(Wo)}; 参数数={Wo.numel()}")
    print(f"四张投影权重共 {sum(w.numel() for w in (Wq, Wk, Wv, Wo))} 个参数")
    print(f"K[0,0,0]={K[0, 0, 0].item():.6f}; "
          f"a[0,0] @ Wk[0]={(a[0, 0] @ Wk[0]).item():.6f}")
    print(f"V[0,0,0]={V[0, 0, 0].item():.6f}")
    for name, tensor in (
        ("Q 分头", q_split), ("K 分头", k_split), ("V 分头", v_split),
        ("Q 旋转后", q_rope), ("K 旋转后", k_rope),
        ("Q 计算布局", q_heads), ("K 重复后布局", k_heads),
        ("V 重复后布局", v_heads), ("scores", scores),
        ("attention_weights", attention_weights), ("attended", attended),
        ("merged", merged), ("o_proj 输出", captured["attention_output"]),
    ):
        print(f"{name}: {shape(tensor)}")
    print("Q 头 0/1/2/3 共用 KV 头 0；Q 头 4/5/6/7 共用 KV 头 1")
    print("第 0 头、第 0 query 的前五个注意力权重:",
          attention_weights[0, 0, 0, :5].tolist())
    print("显式的因果 Attention 计算与真实 o_proj 输入一致。")
    if args.breakpoints:
        breakpoint()

    h = result.last_hidden_state
    logits = result.logits
    torch.testing.assert_close(h, model.model.norm(captured["block1"]))
    torch.testing.assert_close(h, captured["head_input"], rtol=0, atol=0)
    torch.testing.assert_close(h.reshape(N, H) @ E.T, logits.reshape(N, len(tokenizer)))
    torch.testing.assert_close(h[0, 0] @ E[target_id], logits[0, 0, target_id])
    assert tuple(logits.shape) == (1, 63, 6400)
    probabilities = F.softmax(logits[0, 0], dim=-1)
    per_token_loss = F.cross_entropy(
        logits.flatten(0, 1), Y.flatten(), reduction="none",
    ).view_as(Y)
    loss = (per_token_loss * loss_mask).sum() / loss_mask.sum() + result.aux_loss
    torch.testing.assert_close(-probabilities[target_id].log(), per_token_loss[0, 0])
    last_only = model(X, logits_to_keep=1).logits
    torch.testing.assert_close(last_only, logits[:, -1:, :])
    predicted_id = logits[0, 0].argmax().item()

    print("\n[3/3] 两个 Block + 最终 RMSNorm -> lm_head")
    print(f"Block 0: {shape(captured['block0'])}; Block 1: {shape(captured['block1'])}")
    print(f"最终隐藏表示 h: {shape(h)}")
    print(f"logits: ({N} x {H}) @ ({shape(E.T)}) = {N} x {len(tokenizer)} "
          f"-> {shape(logits)}")
    print(f"目标 ID={target_id}, logits[0,0,{target_id}]="
          f"{logits[0, 0, target_id].item():.6f}, "
          f"概率={probabilities[target_id].item():.8f}")
    print(f"最大分数 ID={predicted_id}, token={tokenizer.decode([predicted_id])!r}, "
          f"分数={logits[0, 0, predicted_id].item():.6f}")
    print(f"第 0 个位置 loss={per_token_loss[0, 0].item():.6f}; "
          f"{loss_mask.sum().item()} 个有效目标的 masked loss={loss.item():.6f}")
    print(f"logits.argmax(-1): {shape(logits.argmax(-1))}; "
          f"logits_to_keep=1: {shape(last_only)}")
    print("全部查表、矩阵乘法与 Attention 核对通过；没有写入模型文件。")
    if args.breakpoints:
        breakpoint()
    return {
        "embedding_shape": tuple(embedding.shape),
        "kv_projection_shape": tuple(K.shape),
        "logits_shape": tuple(logits.shape),
        "shared_weight": E is model.lm_head.weight,
        "loss": loss.item(),
    }


if __name__ == "__main__":
    main()
