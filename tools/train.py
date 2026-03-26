import os
import sys
sys.path.append('/root/autodl-tmp/adatad')  # 项目根目录

sys.path.append('/root/autodl-tmp/adatad/opentad/models/utils/post_processing/nms')
sys.path.append('/root/autodl-tmp/adatad/opentad/models/roi_heads/roi_extractors/align1d')
sys.path.append('/root/autodl-tmp/adatad/opentad/models/roi_heads/roi_extractors/boundary_pooling')
sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import time
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)

torch.cuda.empty_cache()
# ------------------------------
# 通用 shape 提取工具
# ------------------------------
def shape_str(t):
    if isinstance(t, torch.Tensor):
        return str(list(t.shape))
    elif isinstance(t, (list, tuple)):
        return "[" + ", ".join(shape_str(x) for x in t) + "]"
    elif isinstance(t, dict):
        return "{" + ", ".join(f"{k}: {shape_str(v)}" for k, v in t.items()) + "}"
    else:
        return str(type(t))

# ------------------------------
# 新增：打印优化器参数组信息的函数
# ------------------------------
def print_optimizer_param_groups(optimizer, logger):
    """
    打印优化器各参数组的 lr、weight_decay 和参数数量
    Args:
        optimizer: 初始化后的优化器
        logger: 日志记录器（保证输出格式统一）
    """
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    for idx, group in enumerate(optimizer.param_groups):
        # 统计该参数组的参数总数
        param_count = 0
        for param in group["params"]:
            param_count += param.numel()  # numel() 统计参数元素个数
        
        # 按指定格式输出（兼容logger和控制台）
        log_msg = (
            f"{current_time} Train INFO: Group {idx}: "
            f"lr={group['lr']:.6f}, weight_decay={group['weight_decay']:.6f}, param_count={param_count}"
        )
        logger.info(log_msg)  # 写入日志文件
        print(log_msg)       # 控制台打印

# ------------------------------
# projection hook (特征投影)
# ------------------------------
import torch

def projection_hook(name):
    def hook(module, input, output):
        print(f"\n🔹 [{name}] (Conv1DTransformerProj)")

        # ---- 打印所有输入形状 ----
        def shape_repr(x):
            if isinstance(x, torch.Tensor):
                return list(x.shape)
            elif isinstance(x, (list, tuple)):
                return [shape_repr(i) for i in x]
            elif isinstance(x, dict):
                return {k: shape_repr(v) for k, v in x.items()}
            else:
                return str(type(x))

        print("  Input shapes:")
        if isinstance(input, (list, tuple)):
            for i, inp in enumerate(input):
                print(f"    Arg{i}: {shape_repr(inp)}")
        else:
            print(f"    {shape_repr(input)}")

        # ---- 打印输出形状 ----
        if isinstance(output, (list, tuple)) and len(output) == 2:
            feats, masks = output
            print("  Output feats:")
            for i, f in enumerate(feats):
                print(f"    Level {i}: {list(f.shape)}")
            print("  Output masks:")
            for i, m in enumerate(masks):
                print(f"    Level {i}: {list(m.shape)}")
        else:
            print(f"  Output: {shape_repr(output)}")

    return hook

# ------------------------------
# neck hook (FPNIdentity)
# ------------------------------
def neck_hook(name):
    def hook(module, input, output):
        print(f"\n🔹 [{name}] (Neck / FPNIdentity)")
        print(f"  Input: {shape_str(input)}")
        print(f"  Output: {shape_str(output)}")
    return hook

# ------------------------------
# rpn_head hook (ActionFormerHead)
# ------------------------------
def rpn_head_hook(name):
    def hook(module, input, output):
        print(f"\n🔹 [{name}] (rpn_head)")
        print(f"  Input: {shape_str(input)}")
        print(f"  Output: {shape_str(output)}")
    return hook

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )

    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.val,
    )

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # build model
    model = build_detector(cfg.model)
    
    if args.rank == 0:
        from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
        from opentad.models.dense_heads.actionformer_head import ActionFormerHead
        # 注册到 projection 层
        for name, module in model.named_modules():
            if name == "projection":
                module.register_forward_hook(projection_hook(name))
                print(f"✅ Registered hook on: {name}")
            elif name == "neck":
                module.register_forward_hook(neck_hook(name))
                print(f"✅ Registered hook on: {name}")
            elif name == "rpn_head":
                module.register_forward_hook(rpn_head_hook(name))
                print(f"✅ Registered hook on: {name}")
            elif "head" in name.lower():
                print(name, type(module))

    # DDP
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False if use_static_graph else True,
        static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
    )
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    
    # ------------------------------
    # 关键新增：打印优化器参数组信息
    # ------------------------------
    if args.rank == 0:  # 仅主进程打印，避免多GPU重复输出
        logger.info("\n========== Optimizer Parameter Groups Info ==========")
        print_optimizer_param_groups(optimizer, logger)
        logger.info("=====================================================\n")
    
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume != None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema != None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")
    val_loss_best = 1e6
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    for epoch in range(resume_epoch + 1, max_epoch):
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
        )

        # save checkpoint
        if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
            if args.rank == 0:
                save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

        # val for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
                val_loss = val_one_epoch(
                    val_loader,
                    model,
                    logger,
                    args.rank,
                    epoch,
                    model_ema=model_ema,
                    use_amp=use_amp,
                )

                # save the best checkpoint
                if val_loss < val_loss_best:
                    logger.info(f"New best epoch {epoch}")
                    val_loss_best = val_loss
                    if args.rank == 0:
                        save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)

        # eval for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
                eval_one_epoch(
                    test_loader,
                    model,
                    cfg,
                    logger,
                    args.rank,
                    model_ema=model_ema,
                    use_amp=use_amp,
                    world_size=args.world_size,
                    not_eval=args.not_eval,
                    
                )
    logger.info("Training Over...\n")

if __name__ == "__main__":
    main()