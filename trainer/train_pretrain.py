"""MiniMind 预训练入口：准备训练对象，再逐轮调用 train_epoch。

源码阅读顺序：先看文件底部的 __main__，再进入 train_epoch，
最后按需查看 train_utils 和 PretrainDataset。
逐段说明见 doc/0.预训练循环精讲.md。
"""

import os
import sys

# 声明当前脚本属于 trainer 包，规范包结构；允许 import trainer.xxx 的导入方式
__package__ = "trainer"
# 把项目根目录加入 Python 模块搜索路径，让脚本能导入上级目录的资源。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.train_utils import (
    get_lr, init_distributed_mode, setup_seed, lm_checkpoint, is_main_process,
    init_model, SkipBatchSampler, Logger, autocast_context, grad_scaler,
    project_path, accumulation_divisor,
)

warnings.filterwarnings('ignore')
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    处理本轮尚未完成的 batch；max_steps 可以限制本轮的处理范围。

    参数:
        epoch: 当前是第几个 epoch（从 0 开始）
        loader: 本次需要读取的数据；恢复时由调用方提前跳过已完成的批次
        iters: 本轮总 batch 数，恢复时等于 len(loader) + start_step
        start_step: 本轮已完成的 batch 数，不是优化器更新次数
        wandb: 可选的实验记录对象；本仓库实际使用 SwanLab

    args、model、lm_config、optimizer、scaler、autocast_ctx 来自本模块
    __main__ 初始化的全局变量。函数每次调用时复用模型及优化器，
    不会重新初始化权重；因此多轮训练能够保留上轮的学习结果。
    """
    # 例如数据可组成 6 批，max_steps=3，则本轮只处理前 3 批。
    # max_steps=0 表示不限制；这里的 step 始终按 batch 计数。
    iters = min(iters, args.max_steps) if args.max_steps else iters
    loss_fct = nn.CrossEntropyLoss(reduction='none') # 不自动求 mean，方便后面用 loss_mask 加权
    start_time = time.time()  # 墙钟时间也包含调试断点的暂停时长

    # 第一次从 batch 1 编号；恢复时从已完成的下一批编号。
    # start 只改变编号，不会跳数据；跳过数据由 SkipBatchSampler 完成。
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        if step > iters:
            break
        # 输入、目标和 mask 必须与模型位于同一设备。
        # X/Y 是 token ID（整数），不会因为启用 AMP 就变成浮点数。
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # ==================== 动态设置当前学习率 ====================
        # 按累计 batch 进度调度学习率，不是按优化器更新次数调度。
        # 累积期间每批都改 lr，真正更新使用该累积组最后一批的 lr。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 把学习率应用到优化器的所有参数组（支持不同层不同 lr）
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ==================== 混合精度前向 ====================
        # CPU/float32 时是空上下文，保持普通精度；CUDA 混合精度时
        # autocast 才按算子选择计算精度，不等于将所有参数转换成半精度。
        with autocast_ctx:
            res = model(X)  # 结构化输出包含 logits 和 aux_loss
            # [B,T,V] -> [B*T,V]，每个预测位置作为一道 V 类选择题；
            # Y 同样展开为 [B*T]，算完再恢复为 [B,T] 的逐位置 loss。
            # cross_entropy 接收原始 logits，不要预先做 softmax。
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # 当前预训练 mask 忽略 padding，并按有效目标数求均值。
            # clamp_min(1) 防止分母为 0，但不代表空监督样本是有效训练数据。
            loss = (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1)
            # 稠密模型的 aux_loss 为 0；MoE 可附加路由负载均衡损失。
            loss += res.aux_loss
            # 按累积组的实际批数缩放：3 批、每 2 批更新时，除数为 2/2/1。
            # backward 会累加 .grad，因此前两批共同形成 (g1+g2)/2。
            divisor = accumulation_divisor(step, iters, args.accumulation_steps)
            loss = loss / divisor

        # ==================== 反向传播 ====================
        # 反向传播只累加梯度，不修改权重。本仓库仅 CUDA float16 启用
        # 梯度缩放；当前 CPU/float32 下等同于 loss.backward()。
        scaler.scale(loss).backward()

        # ==================== 梯度累积到够一步才更新 ====================
        if step % args.accumulation_steps == 0 or step == iters:
            # 1. 若启用了缩放，先还原梯度；否则这是空操作。
            scaler.unscale_(optimizer)
            # 2. 对完整累积组的总梯度范数做裁剪，控制一次更新的梯度幅度。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 3. AdamW 使用梯度及历史状态更新参数；缩放启用时，
            # 非有限梯度可能导致本次更新被跳过，并由 update 调整缩放因子。
            scaler.step(optimizer)
            scaler.update()

            # 4. 更新完成后才清空梯度，避免丢失累积组内前几批的贡献。
            optimizer.zero_grad(set_to_none=True)
            # 5. 释放 CUDA 未占用的缓存块，不会释放仍在使用的模型张量；
            # CPU 本课不需要它，这里保留原训练入口的调用。
            torch.cuda.empty_cache()

        # ==================== 日志打印 & wandb ====================
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 还原累积缩放前的当前 batch loss，便于比较日志。
            # 这是刚才前向算出的数，不是更新后重新测得的 loss，也不是累计均值。
            current_loss = loss.item() * divisor
            current_lr = optimizer.param_groups[-1]['lr']
            # 原实现的粗略分钟估计：含断点暂停，且 step+1 和分别向下
            # 取整会带来偏差；不要把它当作纯计算耗时或可靠的 ETA。
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # ==================== 保存检查点 ====================
        if ((step % args.save_interval == 0 and step % args.accumulation_steps == 0)
                or step == iters) and is_main_process():
            # 保存点位于更新边界，避免恢复时丢掉未写入 checkpoint 的累积梯度。
            # eval 只切换模块运行模式；单纯保存 state_dict 不要求 eval，
            # 它也不会关闭 autograd。下面保留原入口的切换习惯。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # DDP 模型要保存 module.state_dict()
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            # 转换的是待保存的张量，不把内存中的训练参数改成 float16。
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            # 续训文件保存模型、optimizer、scaler、epoch/batch 进度等；
            # 当前实现没有保存模型 config、随机数状态和未更新的累积梯度。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=args.checkpoint_dir)
            model.train()

if __name__ == "__main__":
    # 通过 python -m trainer.train_pretrain 启动时，先进入这里；
    # 上方 def train_epoch 只是定义函数，此时尚未执行训练。
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="out", help="模型保存目录")
    parser.add_argument("--checkpoint_dir", type=project_path, default="checkpoints", help="断点续训目录")
    parser.add_argument("--max_steps", type=int, default=0, help="每轮最多训练的 batch 数，0=不限")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    # zero 是指：跑一遍进行流程验证、快速试错、初步效果评估
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载子进程数，0 表示在当前进程加载")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=1, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    args = parser.parse_args()
    # 相对数据/权重路径按项目根目录解释，而不是依赖 VS Code 的当前目录。
    args.save_dir = project_path(args.save_dir)
    args.data_path = project_path(args.data_path)

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练模式，获取当前进程的「本地 GPU 编号（local_rank）」。如果是「单机单卡训练」（非分布式），这个函数通常会返回 0，且不初始化进程组。
    local_rank = init_distributed_mode()
    # 如果启用分布式训练，每个进程绑定到对应的 GPU 上。
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 固定随机种子，同时保证分布式训练中「各进程的种子不同但固定」（避免多进程重复计算）。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.checkpoint_dir) if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = getattr(torch, args.dtype)
    autocast_ctx = autocast_context(device_type, dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    # 模型只在外层初始化一次。from_weight=none 随机初始化；
    # 指定权重时先加载它，若后面有 resume，再用续训状态覆盖。
    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.save_dir, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 单卡也用一个副本的 DistributedSampler，固定 seed + epoch 的取样顺序，
    # 便于恢复时先复现这一轮顺序，再跳过已完成的 batch。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else DistributedSampler(train_ds, num_replicas=1, rank=0, seed=42)
    # PyTorch 提供的梯度缩放器（Gradient Scaler），主要用于混合精度训练（AMP）
    scaler = grad_scaler(device_type, dtype)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    # 只加载模型权重不够：AdamW 的历史状态和已处理批数也需要恢复。
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)  # 不同轮使用不同但可复现的打乱顺序
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点，首次进入续训，跳过已完成的 step
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=device_type == "cuda")
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # loader 只剩待处理批次；加回 start_step 得到本轮总范围。
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)
