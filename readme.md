# minimind-learn

看到项目 minimind 项目，我大为欣喜，基础薄弱我的决定从头复现一遍该项目。看到网上的教程大多是跑通一遍流程，但是我想完全复刻一遍该项目，学习实现细节。

再次感谢作者的伟大贡献，源项目地址：【[minimind](https://github.com/jingyaogong/minimind)】【[minimind-v](https://github.com/jingyaogong/minimind-v)】

以下是个人复现过程中，的一些记录。包含对于项目内容的理解，一些基础知识补充，以及实验结果。

### 项目文档目录

以下按推荐顺序排列，每个章节逐步深入。

先打开 [学习路线](doc/0.学习路线.md)：它把各章接成“分词 → 数据 → 模型 → 训练 → 不同扩展”的路径，并给出每章小实验和验收点。陌生名词查 [概念速查](doc/0.概念速查.md)。

每章先读场景和图解，再运行本地实验，最后回到源码设断点。可在 VS Code 中打开 Markdown 预览，图表和代码并排看。

```bash
.venv/bin/python scripts/lesson_lab.py --list
.venv/bin/python scripts/lesson_lab.py --lesson sft
.venv/bin/python scripts/lesson_lab.py --lesson all
.venv/bin/python scripts/check_course.py
```

13 个 CPU 小实验使用仓库本地资源；预训练与恢复实验的临时权重自动清理。新增 `--lesson resume` 会对照中断恢复与连续训练的最终状态。PPO/GRPO/VLM 的小算例用于理解关键计算，不代表外部模型依赖和完整训练已经验证。

实践卡住时按 [实践排错](doc/0.实践排错.md) 定位阶段；`check_course.py` 检查本地文档链接、Python 代码块语法和调试入口，修改课程后可重复运行。

如果目标是先在一天内跑通代码，请先看 [一天跑通教程 →](doc/0.一天跑通.md)。它使用当前 ARM64/CUDA 环境和仓库自带的最小数据完成 smoke test、预训练和推理，再回到下面的章节扩展数据和训练阶段。

开始读源码时，按 [逐步调试教程](doc/0.逐步调试.md) 跟踪一个 batch；每步配有学习目标、直观比喻、原理图解和验收练习，配套脚本展示数据、前向、梯度和参数更新，也提供 VS Code 调试配置。

完成三批梯度累积实验后，阅读 [预训练循环精讲](doc/0.预训练循环精讲.md)，从入口准备追踪到 `train_epoch()`，对照实测日志理解每段代码为什么这样写。

理解 gate/up 的参数和矩阵乘法时，查看 [FFN 矩阵与梯度](doc/3.1.FFN矩阵与梯度.md)。用 `.venv/bin/python scripts/inspect_ffn.py` 可从当前真实批次逐步核对 128→384→128 的行列变化、反向梯度和参数更新；VS Code 对应第 8 项配置。

继续理解 Embedding `[1,63,128]`、K/V `[1,63,32]`、logits `[1,63,6400]` 时，查看 [Embedding 到 logits 矩阵精讲](doc/3.2.Embedding到logits矩阵精讲.md)。用 `.venv/bin/python scripts/inspect_model.py` 核对真实查表、8 个 Q 头与 2 个 KV 头、词表点积；VS Code 对应第 9 项配置。

| 序号       | 章节标题              | 查看文档                              |
| ---------- | --------------------- | ------------------------------------- |
| 📚 **0**   | 环境搭建              | [查看文档 →](doc/0.环境搭建.md)       |
| 🗂️ **1**   | train tokenizer       | [查看文档 →](doc/1.train_tokenizer.md)|
| 📦 **2**   | DataLoader            | [查看文档 →](doc/2.DataLoader.md)     |
| 🏗️ **3**   | 模型构建              | [查看文档 →](doc/3.模型构建.md)       |
| 🚀 **4**   | Pretrain              | [查看文档 →](doc/4.Pretrain.md)       |
| 🧑‍🏫 **5**   | SFT                   | [查看文档 →](doc/5.SFT.md)            |
| ⚡ **6**   | LoRA                  | [查看文档 →](doc/6.LoRA.md)           |
| 🍅 | 强化学习实现之☞刨根问底 | [查看文档 →](doc/13.强化学习刨根问底.md) |
| 🏆 **7**   | PPO                   | [查看文档 →](doc/7.PPO.md)            |
| ✅ **8**   | DPO                   | [查看文档 →](doc/8.DPO.md)            |
| 🔬 **9**   | 白盒蒸馏              | [查看文档 →](doc/9.白盒蒸馏.md)       |
| 🧩 **10**  | MoE                   | [查看文档 →](doc/10.MoE.md)           |
| 🔄 **11**  | GRPO                  | [查看文档 →](doc/11.GRPO.md)          |
| 👀 **12**  | minimind-v            | [查看文档 →](doc/12.minimind-v.md)    |

---

> 先按学习路线完成 tokenizer、数据、模型、预训练和 SFT，再按任务选择 LoRA、偏好优化、蒸馏或结构扩展；进阶章节不要求按编号全部串行训练。
