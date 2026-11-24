"""Run small CPU exercises alongside the chapters in doc/."""

import argparse
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset, SFTDataset
from model.model_lora import apply_lora
from model.model_minimind import (
    MiniMindConfig, MiniMindForCausalLM, MoEGate, MoEFeedForward, RMSNorm,
)
from model.model_vlm import VisionProj, VLMConfig
from trainer.train_dpo import dpo_loss
from trainer.train_distillation import distillation_loss
from trainer.train_ppo import GAE
from trainer.train_utils import setup_seed, torch_load


def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT / "model")


def small_model():
    return MiniMindForCausalLM(MiniMindConfig(
        hidden_size=128, intermediate_size=384, num_hidden_layers=2,
        max_position_embeddings=128,
    ))


def lesson_tokenizer():
    tok = tokenizer()
    text = "MiniMind"
    ids = tok(text, add_special_tokens=False).input_ids
    print("文本:", text)
    print("token ID:", ids)
    print("逐片解码:", [tok.decode([item]) for item in ids])
    assert ids == [47, 261, 75, 47, 807]
    assert tok.decode(ids) == text
    print("词表大小:", len(tok), "；ID 是编号，不是分数或向量。")


def lesson_data():
    tok = tokenizer()
    dataset = PretrainDataset(ROOT / "dataset/demo_pretrain.jsonl", tok, max_length=64)
    X, Y, mask = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert X.shape == Y.shape == mask.shape == (2, 63)
    assert torch.equal(X[:, 1:], Y[:, :-1])
    print("单条样本形状:", tuple(dataset[0][0].shape))
    print("两条组批形状:", tuple(X.shape))
    for position in range(5):
        print(position, repr(tok.decode([X[0, position].item()])),
              "->", repr(tok.decode([Y[0, position].item()])),
              "mask=", mask[0, position].item())
    print("第 0 条有效目标数:", mask[0].sum().item())


def lesson_model():
    from scripts.debug_train_step import main
    metrics = main(["--stop_after", "forward", "--token_rows", "0"])
    assert metrics["logits_shape"] == (1, 63, 6400)
    normalized = RMSNorm(2)(torch.tensor([[3.0, 4.0]]))
    activated = F.silu(torch.tensor([-1.0, 0.0, 2.0]))
    print("RMSNorm([3,4]):", normalized.detach().tolist())
    print("SiLU([-1,0,2]):", activated.tolist(), "；SiLU 不是 0～1 的概率门。")
    assert activated[0] < 0 and activated[2] > 1


def lesson_pretrain():
    # Exercise the real entry point; temporary weights do not overwrite out/.
    with tempfile.TemporaryDirectory(prefix="minimind_lesson_") as directory:
        arguments = [
            "trainer.train_pretrain", "--device", "cpu", "--dtype", "float32",
            "--hidden_size", "128", "--num_hidden_layers", "2",
            "--max_seq_len", "64", "--batch_size", "1", "--num_workers", "0",
            "--epochs", "1", "--max_steps", "3", "--accumulation_steps", "2",
            "--from_weight", "none", "--from_resume", "0",
            "--log_interval", "1", "--save_interval", "2",
            "--data_path", str(ROOT / "dataset/demo_pretrain.jsonl"),
            "--save_dir", str(Path(directory) / "out"),
            "--checkpoint_dir", str(Path(directory) / "checkpoints"),
        ]
        with patch.object(sys, "argv", arguments):
            runpy.run_module("trainer.train_pretrain", run_name="__main__")
        checkpoint = torch_load(Path(directory) / "checkpoints/pretrain_128_resume.pth")
        counts = {int(state["step"].item())
                  for state in checkpoint["optimizer"]["state"].values()}
        assert checkpoint["step"] == 3 and counts == {2}
        print("真实入口验收：处理 3 批，AdamW 更新 2 次；临时权重随后自动清理。")


def lesson_sft():
    tok = tokenizer()
    dataset = SFTDataset(ROOT / "dataset/demo_sft.jsonl", tok, max_length=128)
    X, Y, mask = dataset[0]
    supervised = int(mask.sum().item())
    nonpadding = int((Y != tok.pad_token_id).sum().item())
    print("模板文本:\n" + dataset._create_chat_prompt(dataset.samples[0]["conversations"]))
    print("参与计分的 Y:\n" + tok.decode(Y[mask.bool()].tolist()))
    print("有效监督 / 非 padding 目标:", supervised, "/", nonpadding)
    assert supervised == 21 and nonpadding == 52
    assert torch.equal(X[1:], Y[:-1])
    print("用户问题仍在 X 中；mask=0 只是不为那个目标位置计分。")


def lesson_lora():
    tok = tokenizer()
    model = small_model()
    dataset = PretrainDataset(ROOT / "dataset/demo_pretrain.jsonl", tok, max_length=64)
    X, Y, mask = next(iter(DataLoader(Subset(dataset, [0]), batch_size=1)))
    with torch.no_grad():
        original = model(X).logits.clone()
    apply_lora(model)
    with torch.no_grad():
        torch.testing.assert_close(original, model(X).logits)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora" in name)
    base = model.model.layers[0].self_attn.q_proj
    before = base.weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3,
    )
    logits = model(X).logits
    per_token = F.cross_entropy(logits.flatten(0, 1), Y.flatten(), reduction="none").view_as(Y)
    loss = (per_token * mask).sum() / mask.sum()
    loss.backward()
    assert torch.count_nonzero(base.lora.A.weight.grad).item() == 0
    assert base.lora.B.weight.grad.norm().item() > 0
    optimizer.step()
    torch.testing.assert_close(base.weight, before)
    assert torch.count_nonzero(base.lora.B.weight).item() > 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("注入前后初始输出一致；基础 W 更新量为 0；LoRA B 已更新。")
    print("小模型 LoRA 可训练参数:", trainable)
    print("首轮 A 梯度为 0 是 B 零初始化的链式求导结果，不代表 A 被冻结。")


def lesson_dpo():
    reference = torch.tensor([[-2.0], [-2.0]])
    policy = torch.tensor([[-1.0], [-2.0]], requires_grad=True)
    mask = torch.ones(2, 1)
    good = dpo_loss(reference, policy, mask, beta=0.1)
    bad = dpo_loss(reference, torch.tensor([[-3.0], [-2.0]]), mask, beta=0.1)
    good.backward()
    assert good < bad
    assert policy.grad[0, 0] < 0 and policy.grad[1, 0] > 0
    print(f"相对参考模型更偏好 chosen：loss={good.item():.6f}")
    print(f"相对参考模型更偏好 rejected：loss={bad.item():.6f}")
    print("policy 对数概率的梯度:", policy.grad.flatten().tolist())
    print("调用的是仓库 dpo_loss，前半 batch 为 chosen、后半为 rejected。")


def lesson_distill():
    teacher = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
    student = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    equal = distillation_loss(teacher.clone(), teacher, temperature=2.0)
    different = distillation_loss(student, teacher, temperature=2.0)
    different.backward()
    torch.testing.assert_close(equal, torch.tensor(0.0), atol=1e-6, rtol=0)
    assert different > equal and student.grad.norm() > 0
    assert teacher.grad is None
    print("教师软概率:", (teacher / 2).softmax(-1).tolist())
    print(f"同分布 KD loss={equal.item():.6f}；反向分布 KD loss={different.item():.6f}")
    print("学生有梯度，教师无梯度；本实验无需加载教师权重。")


def lesson_moe():
    config = MiniMindConfig(hidden_size=4, intermediate_size=8,
                            n_routed_experts=4, num_experts_per_tok=2)
    gate = MoEGate(config)
    with torch.no_grad():
        gate.weight.copy_(torch.eye(4))
    scores = torch.tensor([0.1, 0.6, 0.2, 0.1])
    indices, weights, _ = gate(scores.log().view(1, 1, 4))
    mapping = dict(zip(indices[0].tolist(), weights[0].tolist()))
    assert set(mapping) == {1, 2}
    assert abs(mapping[1] - 0.75) < 1e-6 and abs(mapping[2] - 0.25) < 1e-6
    layer = MoEFeedForward(config)
    inputs = torch.randn(2, 3, 4)
    with torch.no_grad():
        training_output = layer.train()(inputs)
        inference_output = layer.eval()(inputs)
    torch.testing.assert_close(training_output, inference_output, atol=1e-6, rtol=1e-5)
    print("门控原始概率:", scores.tolist())
    print("选中专家及归一化权重:", mapping)
    print("真实 MoE 两条计算路径输出一致，形状:", tuple(inference_output.shape))


def lesson_ppo():
    rewards = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    values = np.array([[0.2, 0.3, 0.4]], dtype=np.float32)
    done = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    advantages = GAE(1, 3, gamma=1.0, lambda_=1.0)(done, rewards, values)
    np.testing.assert_allclose(advantages, [[0.8, 0.7, 0.6]], atol=1e-6)
    ratio = torch.tensor(1.5, requires_grad=True)
    clipped_loss = -torch.minimum(ratio, ratio.clamp(0.8, 1.2))
    clipped_loss.backward()
    assert abs(clipped_loss.item() + 1.2) < 1e-6 and ratio.grad.item() == 0
    print("真实 GAE 函数，终点奖励为 1：advantages =", advantages.tolist())
    print("对应 returns:", (advantages + values).tolist())
    print("正优势为 1、ratio=1.5、epsilon=0.2：裁剪目标为 1.2，该侧梯度为 0。")
    print("这是 GAE 与损失算例，不代表完整 PPO 或奖励模型已经跑通。")


def lesson_grpo():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    advantages = (rewards - rewards.mean(1, keepdim=True)) / (
        rewards.std(1, keepdim=True) + 1e-4)
    assert advantages[0, 0] < 0 and advantages[0, 1] == 0 and advantages[0, 2] > 0
    logp = torch.tensor([-0.5, -0.8], requires_grad=True)
    ratio = torch.exp(logp - logp.detach())
    torch.testing.assert_close(ratio, torch.ones_like(ratio))
    ratio.sum().backward()
    torch.testing.assert_close(logp.grad, torch.ones_like(logp))
    print("同一问题三份回答的奖励:", rewards.tolist())
    print("组内优势:", advantages.tolist())
    print("exp(logp - logp.detach()) 的值:", ratio.detach().tolist())
    print("对 logp 的梯度:", logp.grad.tolist(), "；值为 1 不代表没有梯度。")
    print("这是当前脚本关键表达式的数学练习，不加载奖励模型。")


def lesson_vlm():
    tok = tokenizer()
    image_ids = tok("@" * 196, add_special_tokens=False).input_ids
    assert image_ids == VLMConfig().image_ids
    projector = VisionProj(ve_hidden_size=768, hidden_size=128)
    patches = torch.randn(1, 196, 768)
    projected = projector(patches)
    combined = torch.cat([torch.zeros(1, 20, 128), projected,
                          torch.zeros(1, 10, 128)], dim=1)
    assert combined.shape == (1, 226, 128)
    combined.square().mean().backward()
    assert projector.vision_proj[0].weight.grad.norm().item() > 0
    print("占位 token 数:", len(image_ids))
    print("真实 VisionProj:", tuple(patches.shape), "->", tuple(projected.shape))
    print("20 个文本位置 + 196 个图像位置 + 10 个文本位置:", tuple(combined.shape))
    print("这里只验证投影、位置拼接与梯度；未下载 CLIP，未做真实图像问答。")


LESSONS = {
    "tokenizer": ("切片与编号", lesson_tokenizer),
    "data": ("一条样本如何组批", lesson_data),
    "model": ("前向形状与数值", lesson_model),
    "pretrain": ("真实入口：三批、两次更新", lesson_pretrain),
    "sft": ("题干可读，回答计分", lesson_sft),
    "lora": ("冻结基础权重，更新低秩分支", lesson_lora),
    "dpo": ("成对回答的偏好损失", lesson_dpo),
    "distill": ("模仿教师的软概率", lesson_distill),
    "moe": ("真实门控与专家融合", lesson_moe),
    "ppo": ("终点奖励、GAE 与裁剪", lesson_ppo),
    "grpo": ("组内比较与 detach 的梯度", lesson_grpo),
    "vlm": ("图像占位与真实投影层", lesson_vlm),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", choices=[*LESSONS, "all"], default="data")
    parser.add_argument("--list", action="store_true", help="List available exercises")
    args = parser.parse_args()
    if args.list:
        for name, (description, _) in LESSONS.items():
            print(f"{name:10} {description}")
        return
    torch.set_num_threads(2)
    selected = list(LESSONS) if args.lesson == "all" else [args.lesson]
    for name in selected:
        description, exercise = LESSONS[name]
        setup_seed(42)
        print(f"\n[{name}] {description}", flush=True)
        exercise()
        print(f"PASS {name}", flush=True)
    print(f"\n完成 {len(selected)} 个实验；通过表示本实验的检查点满足，不是模型能力评分。")


if __name__ == "__main__":
    main()
