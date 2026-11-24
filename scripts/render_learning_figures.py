"""Render the learning map and a real SFT supervision table."""

from pathlib import Path
import sys

from transformers import AutoTokenizer

from render_pretrain_figures import BLUE, GOLD, GREEN, INK, MUTED, RED, box, canvas, label, lines

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import SFTDataset


def learning_map():
    image, draw = canvas(
        "同一台接龙机器，后续章节分别换哪里？",
        "先走通文字到参数更新，再按任务选择扩展；LoRA、MoE 和 SFT 可以组合。",
        1150,
    )
    box(draw, (48, 180, 1392, 340), "#f3f7fd")
    label(draw, (74, 198), "共同主线", 31, BLUE, True)
    lines(draw, (74, 258), [
        "文字 → token ID → X / Y / mask → 模型 → logits → loss → 梯度 → 更新",
    ], size=28, width=1292)
    items = [
        ("换监督任务", BLUE, "Pretrain：正文接龙", "SFT：根据问题学习标准回答"),
        ("换损失信号", GREEN, "DPO：比较 chosen / rejected", "蒸馏：模仿教师的概率分布"),
        ("换反馈来源", RED, "PPO：生成、奖励、Critic 与优势", "GRPO：同题多答，组内比较"),
        ("换可训练范围", GOLD, "LoRA：基础矩阵保持不动", "只训练低秩增量分支 A、B"),
        ("换内部结构", BLUE, "MoE：门控挑选部分专家 FFN", "每个 token 路由后加权融合"),
        ("换输入来源", GREEN, "VLM：CLIP 特征经过投影", "放进文本序列预留的图片位置"),
    ]
    for index, (title, color, first, second) in enumerate(items):
        x = 48 + index % 2 * 688
        y = 374 + index // 2 * 222
        box(draw, (x, y, x + 656, y + 196), "#fafbfc")
        label(draw, (x + 24, y + 15), title, 30, color, True)
        lines(draw, (x + 24, y + 79), [first, second], size=25, width=608)
    label(draw, (48, 1070), "学习方法：先预测 → 跑本地小实验 → 看断点变量 → 换一个输入解释变化。", 27, INK, width=1344)
    return image


def sft_table():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    dataset = SFTDataset(ROOT / "dataset/demo_sft.jsonl", tokenizer, max_length=128)
    X, Y, mask = dataset[0]
    image, draw = canvas(
        "SFT：同一个位置，要对照 X、Y 和 mask 一起看",
        "以下行来自当前仓库第 0 条 SFT 样本；数字为真实 token 位置，换行显示成 \\n。",
        1100,
    )
    label(draw, (48, 181), "用户：What is MiniMind?    助手：MiniMind is a small language model ...", 28, BLUE, True, width=1344)
    columns = [(70, "预测位置"), (250, "X：当前输入"), (580, "Y：下一个目标"), (970, "mask / 是否计分")]
    for x, title in columns:
        label(draw, (x, 250), title, 26, MUTED, True)
    positions = [23, 27, 30, 31, 32, 50, 51, 52]
    for row, position in enumerate(positions):
        y = 314 + row * 72
        supervised = bool(mask[position].item())
        box(draw, (48, y - 5, 1392, y + 58), "#eef8f1" if supervised else "#f4f6f8")
        label(draw, (70, y), str(position), 27)
        label(draw, (250, y), repr(tokenizer.decode([X[position].item()])), 27, BLUE, width=310)
        label(draw, (580, y), repr(tokenizer.decode([Y[position].item()])), 27, GREEN if supervised else MUTED, width=370)
        label(draw, (970, y), "1  计分" if supervised else "0  忽略目标损失", 27, GREEN if supervised else MUTED, width=390)
    lines(draw, (48, 931), [
        "第 31 行：看到 assistant 头后的换行，预测回答首片 M，开始计分。",
        "第 50 / 51 行：结束标记和紧接的换行参与监督；第 52 行 padding 目标不计分。",
        "用户问题仍然在输入中，mask=0 不会删除它，也不会禁止模型用它理解回答。",
    ], size=25, width=1344)
    assert mask[30].item() == 0 and mask[31].item() == 1 and mask[52].item() == 0
    return image


def main():
    for relative, image in [
        ("doc/assets/0.学习路线/01-methods.png", learning_map()),
        ("doc/assets/5.SFT/01-supervision.png", sft_table()),
    ]:
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, optimize=True)
        print(relative)


if __name__ == "__main__":
    main()
