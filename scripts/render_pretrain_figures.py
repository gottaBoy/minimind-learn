"""Render the Chinese diagrams embedded in the pretraining lesson."""

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "doc/assets/0.预训练循环精讲"
FONT_DIRECTORY = Path("/usr/share/fonts/opentype/noto")
INK = "#202b32"
MUTED = "#596770"
LINE = "#d5dde1"
BLUE = "#2267ac"
GREEN = "#22784e"
RED = "#b03d4c"
GOLD = "#946d0b"


def font(size, bold=False):
    weight = "Bold" if bold else "Regular"
    return ImageFont.truetype(str(FONT_DIRECTORY / f"NotoSansCJK-{weight}.ttc"), size)


def label(draw, position, content, size=25, color=INK, bold=False, width=None):
    selected = font(size, bold)
    if width is not None:
        assert draw.textbbox((0, 0), content, font=selected)[2] <= width, content
    draw.text(position, content, font=selected, fill=color)


def lines(draw, position, content, size=25, color=INK, gap=14, width=None):
    x, y = position
    for line in content:
        label(draw, (x, y), line, size, color, width=width)
        y += size + gap


def box(draw, rectangle, fill="#ffffff", outline=LINE):
    draw.rounded_rectangle(rectangle, radius=8, fill=fill, outline=outline, width=2)


def arrow(draw, start, end, color=MUTED):
    draw.line((start, end), fill=color, width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    wings = [
        (end[0] - 16 * math.cos(angle - 0.45), end[1] - 16 * math.sin(angle - 0.45)),
        (end[0] - 16 * math.cos(angle + 0.45), end[1] - 16 * math.sin(angle + 0.45)),
    ]
    draw.polygon([end, *wings], fill=color)


def canvas(title, subtitle, height):
    image = Image.new("RGB", (1440, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    label(draw, (48, 30), title, 40, bold=True, width=1344)
    label(draw, (48, 96), subtitle, 24, MUTED, width=1344)
    draw.line((48, 150, 1392, 150), fill=LINE, width=2)
    return image, draw


def workbench():
    image, draw = canvas(
        "把训练想成一台能调整旋钮的接龙机器",
        "旋钮是模型参数；梯度是暂存的调整建议。只有优化器执行更新，旋钮才会转动。",
        1010,
    )
    box(draw, (48, 182, 1392, 330), "#f3f6f8")
    label(draw, (76, 199), "train_pretrain.py 的 __main__：先把工作台准备好", 29, BLUE, True)
    lines(draw, (76, 253), [
        "读参数  →  固定随机起点  →  创建机器和题库  →  配置调参规则  →  开始一轮",
    ], width=1288)

    box(draw, (48, 380, 418, 620), "#f3f7fd")
    label(draw, (72, 400), "1  题目和答案", 30, BLUE, True)
    lines(draw, (72, 463), ["X：看到 M", "Y：应该接 in", "mask：这道题要计分"], width=322)
    arrow(draw, (422, 500), (482, 500))
    box(draw, (488, 380, 928, 620), "#f1f8f3")
    label(draw, (512, 400), "2  模型里的参数旋钮", 30, GREEN, True)
    lines(draw, (512, 463), [
        "按当前旋钮位置计算分数",
        "forward：先算，不调",
        "logits：每个候选一个分数",
    ], width=392)
    arrow(draw, (933, 500), (990, 500))
    box(draw, (996, 380, 1392, 620), "#fff8e8")
    label(draw, (1020, 400), "3  对照答案算 loss", 30, GOLD, True)
    lines(draw, (1020, 463), [
        "正确项概率低 → loss 大",
        "正确项概率高 → loss 小",
        "loss 是数值，不是梯度",
    ], size=24, width=348)

    arrow(draw, (1194, 625), (1194, 694))
    box(draw, (808, 700, 1392, 870), "#fcf2f3")
    label(draw, (832, 718), "4  backward：把建议记进 .grad", 28, RED, True)
    lines(draw, (832, 776), ["现在有了梯度，但参数旋钮还没转动", "下一批可以把新建议累加到这里"], width=536)
    arrow(draw, (801, 785), (678, 785), GREEN)
    box(draw, (48, 700, 670, 870), "#f1f8f3")
    label(draw, (72, 718), "5  optimizer.step：按规则转动旋钮", 28, GREEN, True)
    lines(draw, (72, 776), ["组内建议收齐 → 裁剪 → 更新参数", "zero_grad：擦掉已用建议，保留新旋钮位置"], size=24, width=574)
    label(draw, (48, 917), "train_epoch()：反复执行这套做题流程；同一台机器会继续保留已经调整的参数。", 26, BLUE, width=1344)
    return image


def next_token(tokenizer):
    image, draw = canvas(
        "跟住第一道题：看到 M，下一片应该是 in",
        "上半图使用当前 tokenizer 的真实切分；下半图把 6400 个候选缩成两个，用教学数字演示。",
        1120,
    )
    ids = tokenizer("MiniMind is a small", add_special_tokens=False).input_ids
    pieces = [repr(tokenizer.decode([token_id])) for token_id in ids]
    label(draw, (48, 180), "同一条文本错开一格，得到一排题目和一排标准答案", 29, BLUE, True)
    cell_width = 160
    for index in range(7):
        x = 250 + index * cell_width
        label(draw, (x + 14, 239), f"位置 {index}", 22, MUTED)
        box(draw, (x, 286, x + 145, 363), "#f3f7fd")
        label(draw, (x + 10, 301), pieces[index], 26, BLUE, True, width=125)
        box(draw, (x, 390, x + 145, 467), "#f1f8f3")
        label(draw, (x + 10, 405), pieces[index + 1], 26, GREEN, True, width=125)
    label(draw, (48, 301), "X  输入", 29, BLUE, True)
    label(draw, (48, 405), "Y  答案", 29, GREEN, True)
    arrow(draw, (322, 368), (322, 385))
    label(draw, (48, 500), "第 0 道题：X[0,0] 是 47（M）；Y[0,0] 是 261（in）。Y 是数据里的答案。", 25, width=1344)

    for x, title, theta, probability, loss in [
        (48, "更新前：旋钮 θ = 0", 0.0, 0.5, math.log(2)),
        (788, "更新后：旋钮 θ = 0.5", 0.5, 1 / (1 + math.exp(-0.5)), math.log1p(math.exp(-0.5))),
    ]:
        box(draw, (x, 571, x + 604, 929), "#f8fafb")
        label(draw, (x + 24, 590), title, 30, GREEN if theta else BLUE, True)
        for row, token, chance, score in [
            (0, "in", probability, theta),
            (1, " a", 1 - probability, 0.0),
        ]:
            y = 661 + row * 78
            label(draw, (x + 24, y), f"{token!r}：{score:.1f} 分", 25)
            draw.rectangle((x + 210, y + 8, x + 430, y + 36), fill="#e3e8eb")
            draw.rectangle((x + 210, y + 8, x + 210 + 220 * chance, y + 36), fill=GREEN if row == 0 else "#8b98a1")
            label(draw, (x + 448, y), f"{chance:.1%}", 25, GREEN if row == 0 else MUTED)
        label(draw, (x + 24, 827), f"正确答案是 in，loss = {loss:.3f}", 26, GREEN, True)
        label(draw, (x + 24, 876), "分数 → softmax → 正确项概率 → loss", 22, MUTED, width=556)
    arrow(draw, (662, 745), (778, 745), RED)
    label(draw, (666, 674), "梯度", 23, RED)
    label(draw, (666, 710), "-0.5", 23, RED)
    lines(draw, (48, 971), [
        "backward 后：θ 仍为 0，只多了 θ.grad = -0.5。",
        "SGD(lr=1).step 后：θ 变为 0.5，正确项概率升高。这里只演示原理，真实训练使用 AdamW。",
    ], size=25, width=1344)
    return image


def accumulation():
    image, draw = canvas(
        "三批数据，两次更新：参数和梯度分开看",
        "loss 是你的真实日志；g1/g2/g3 是每批梯度的符号。先记录反馈，累积组结束才更新。",
        1110,
    )
    columns = [
        ("第 1 批", "W0", "8.892608", "g1 / 2", "保持 W0", "留着这份梯度", BLUE),
        ("第 2 批", "W0", "8.745841", "g1 / 2 + g2 / 2", "更新为 W1", "用完后清空梯度", GREEN),
        ("第 3 批", "W1", "8.861334", "g3 / 1", "更新为 W2", "用完后清空梯度", RED),
    ]
    for index, (batch, weight, loss, gradient, action, cleanup, color) in enumerate(columns):
        x = 48 + index * 456
        box(draw, (x, 184, x + 432, 850), "#fafbfc")
        label(draw, (x + 24, 203), batch, 35, color, True)
        label(draw, (x + 24, 272), "前向时的参数", 24, MUTED)
        label(draw, (x + 24, 311), weight, 43, color, True)
        label(draw, (x + 24, 390), "算出本批 loss", 24, MUTED)
        label(draw, (x + 24, 429), loss, 34, INK, True)
        draw.line((x + 24, 500, x + 408, 500), fill=LINE, width=2)
        label(draw, (x + 24, 522), "backward 后的梯度暂存区", 23, RED)
        label(draw, (x + 24, 568), gradient, 29, RED, True, width=384)
        label(draw, (x + 24, 664), action, 32, color, True)
        label(draw, (x + 24, 725), cleanup, 25)
        label(draw, (x + 24, 786), "不更新" if index == 0 else f"第 {index} 次更新", 24, color)
    box(draw, (48, 889, 1392, 1052), "#f3f6f8")
    lines(draw, (74, 910), [
        "第 1、2 批都用 W0 做题，loss 不同首先是因为题目不同。",
        "第 2 行打印在更新后，但 8.745841 仍是刚才 W0 算出的分数；没有用 W1 重考。",
        "zero_grad 擦掉的是梯度暂存区，不会把参数从 W1 改回 W0。",
    ], size=25, width=1292)
    return image


def main():
    torch.set_num_threads(2)
    knob = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.SGD([knob], lr=1.0)
    scores = torch.stack((knob, torch.tensor(0.0)))
    loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([0]))
    loss.backward()
    assert knob.item() == 0.0 and knob.grad.item() == -0.5
    optimizer.step()
    assert knob.item() == 0.5

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    assert tokenizer("MiniMind", add_special_tokens=False).input_ids == [47, 261, 75, 47, 807]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, image in [
        ("01-workbench.png", workbench()),
        ("02-next-token.png", next_token(tokenizer)),
        ("03-accumulation.png", accumulation()),
    ]:
        destination = OUTPUT / filename
        image.save(destination, optimize=True)
        print(destination.relative_to(ROOT))
    print("Verified: toy backward keeps theta=0; SGD step changes theta to 0.5")


if __name__ == "__main__":
    main()
