"""
This training script can be run both on a single GPU in debug mode,
and also in a larger training run with distributed data parallel (DDP).

Examples
--------
Single GPU:
    python train_baseline.py --batch_size=32 --compile=False

DDP on 4 GPUs, 1 node:
    torchrun --standalone --nproc_per_node=4 train_baseline.py

DDP on 8 GPUs across 2 nodes:
    # master node (example IP 123.456.123.456)
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
        --master_addr=123.456.123.456 --master_port=1234 train_baseline.py

    # worker node
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
        --master_addr=123.456.123.456 --master_port=1234 train_baseline.py

If your cluster does not have Infiniband interconnect, prepend:
    NCCL_IB_DISABLE=1
"""

import os
import time
from contextlib import nullcontext

import argparse
import itertools

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.utils import (
    cosine_scheduler,
    prepare_GPP_dataset,
    prepare_MoBI_dataset,
    prepare_FBM_dataset,
    get_metrics,
)
from utils.myaml import load_config
from models import (
    SPaRCNet,
    ContraWR,
    FFCL,
    STTransformer,
)

# Global state (set in init / main)
master_process = None
device = None
dtype = None
ctx = None
ddp_rank = None
device_type = None
ddp = None
ddp_world_size = None
ddp_local_rank = None

# Global variables defined in __main__
# model_name, log_txt, train_sessions, val_sessions, test_sessions, config
# test_fold, val_fold, target_domains, monitor, metrics


def init(args):
    """
    Initialize device, DDP, seeds, and autocast context.
    """
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    backend = "nccl"
    device = "cuda"
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype
    )


def evaluate(model, dataloader, metrics, save_predntrue: bool = False):
    """
    Run evaluation on a dataloader and return metric logs dict.
    Optionally save prediction/ground-truth arrays (only on rank 0).
    """
    global ctx, model_name, ddp_rank, device

    with torch.no_grad():
        model.eval()
        pred = []
        true = []
        total_loss = []
        log = {}

        for _, batch in enumerate(dataloader):
            # Ensure EEG_X shape matches model expectation
            if model_name.split("-")[0] not in [
                "SPaRCNet",
                "ContraWR",
                "FFCL",
                "STtransformer",
                "EEGConformer",
                "TCN",
            ]:
                if isinstance(batch["EEG_X"], np.ndarray):
                    batch["EEG_X"] = np.expand_dims(batch["EEG_X"], axis=1)
                elif isinstance(batch["EEG_X"], torch.Tensor):
                    batch["EEG_X"] = batch["EEG_X"].unsqueeze(1)

            if model_name.split("-")[0] in ["TCN"]:
                if isinstance(batch["EEG_X"], np.ndarray):
                    batch["EEG_X"] = np.transpose(batch["EEG_X"], (0, 2, 1))
                elif isinstance(batch["EEG_X"], torch.Tensor):
                    batch["EEG_X"] = batch["EEG_X"].permute(0, 2, 1)

            data = {
                "EEG": batch["EEG_X"].float().to(device, non_blocking=True)
                if "EEG_X" in batch
                else None,
                "domain": batch["domain"].to(device, non_blocking=True)
                if "domain" in batch
                else None,
                "Y": batch["Y"].to(device, non_blocking=True)
                if "Y" in batch
                else None,
            }

            with ctx:
                loss, logits = model(data, metrics=metrics)

            pred.append(logits.cpu())
            true.append(data["Y"].cpu())
            total_loss.append(loss)

        pred = torch.cat(pred, dim=0)
        true = torch.cat(true, dim=0)
        total_loss = float(np.mean(total_loss))

        # Save npy only on rank 0 if requested
        if ddp_rank == 0 and save_predntrue:
            tmp = torch.unsqueeze(pred, 1)
            tmp = torch.cat([tmp, torch.unsqueeze(true, 1)], dim=1)
            np.save(f"predntrue_{tmp.shape[0]}.npy", tmp.numpy())
            print(f"Rank 0 saved predntrue_{tmp.shape[0]}.npy")

        # Compute metrics
        if "r2" in metrics:
            results = get_metrics(pred.numpy(), true.numpy(), metrics, is_binary=False)
        elif "f1_weighted" not in metrics:
            results = get_metrics(
                torch.sigmoid(pred).numpy(), true.numpy(), metrics, is_binary=True
            )
        else:
            results = get_metrics(pred.numpy(), true.numpy(), metrics, is_binary=False)

        log["val/total_loss"] = total_loss
        for key, value in results.items():
            log[f"val/{key}"] = value

        model.train()
        return log


def get_model(model_name: str, config):
    """
    Build model by name and move to global device.
    """
    global device, target_domains, args

    if model_name.startswith("EEGConformer"):
        from models.EEGConformer import MyEEGConformer

        model = MyEEGConformer(
            n_outputs=config.eegnet.num_chan_kin,
            n_chans=config.eegnet.num_chan_eeg,
            n_times=config.eegnet.eeg.time_step,
            final_fc_length=840,
            add_log_softmax=False,
        ).to(device)

    elif model_name.startswith("GaitGraph"):
        from models.GaitGraph_model import GaitGraph_tmp

        model = GaitGraph_tmp(config, init_adj=False).to(device)

    elif model_name.startswith("SPaRCNet"):
        model = SPaRCNet(
            in_channels=config.eegnet.num_chan_eeg,
            sample_length=config.eegnet.eeg.time_step,
            n_classes=config.eegnet.num_chan_kin,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        ).to(device)

    elif model_name.startswith("ContraWR"):
        model = ContraWR(
            in_channels=config.eegnet.num_chan_eeg,
            n_classes=config.eegnet.num_chan_kin,
            fft=20,
            steps=2,
        ).to(device)

    elif model_name.startswith("FFCL"):
        model = FFCL(
            in_channels=config.eegnet.num_chan_eeg,
            n_classes=config.eegnet.num_chan_kin,
            fft=20,
            steps=2,
            sample_length=config.eegnet.eeg.time_step,
            shrink_steps=20,
        ).to(device)

    elif model_name.startswith("TSception"):
        from models.TSception_model import TSception

        model = TSception(config).to(device)

    elif model_name.startswith("STtransformer"):
        model = STTransformer(
            n_classes=config.eegnet.num_chan_kin,
            channel_legnth=config.eegnet.eeg.time_step,
            n_channels=config.eegnet.num_chan_eeg,
        ).to(device)

    elif model_name.startswith("EEGNet"):
        from EEGNet import EEGNet

        model = EEGNet(config).to(device)

    elif model_name.startswith("deepConvNet"):
        from models.DCN_model import deepConvNet

        model = deepConvNet(config).to(device)
        if args.pretrained:
            checkpoint = torch.load(args.pretrained_model_path, map_location=device)
            state_dict = checkpoint["model"]
            model_dict = model.state_dict()

            filtered_state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and not k.startswith("lastLayer")
            }

            model_dict.update(filtered_state_dict)
            model.load_state_dict(model_dict)
            print(
                f"Loaded pretrained weights excluding 'lastLayer'. "
                f"Loaded {len(filtered_state_dict)} layers."
            )

    elif model_name.startswith("deepConvNet_sv"):
        from models.DCN_model import deepConvNet_sv

        model = deepConvNet_sv(config).to(device)

    elif model_name.startswith("deepConvNet_multihead"):
        from models.DCN_model import deepConvNet_multihead

        model = deepConvNet_multihead(config, target_domains=target_domains).to(device)

    elif model_name.startswith("deepConvNet_lastspatial"):
        from models.DCN_model import deepConvNet_latespatial

        model = deepConvNet_latespatial(config).to(device)

    elif model_name.startswith("TCN"):
        from models.tcn_model import TCN

        model = TCN(config).to(device)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model



def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank, monitor, metrics

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, args.dataset + "_ind", model_name)
    final_filename = f"ckpt_{args.dataset}_p2_fold{test_fold}_49.pt"

    if not os.path.exists(checkpoint_out_dir):
        os.makedirs(checkpoint_out_dir)

    if os.path.exists(os.path.join(checkpoint_out_dir, final_filename)):
        print(
            f"Final model already exists at "
            f"{os.path.join(checkpoint_out_dir, final_filename)}. Skipping training."
        )
        return

    model = get_model(model_name, config)

    # init_from: 'resume' or 'scratch'
    if args.resume:
        init_from = "resume"
        print("\n\ninit from resume\n\n")
    else:
        init_from = "scratch"
        print("init from scratch\n\n")

    iter_num = 0
    save_filename = f"ckpt_{args.dataset}_p2_fold{test_fold}_{val_fold}"
    save_filename_test = f"ckpt_{args.dataset}_p2_fold{test_fold}"
    save_filename_val = f"ckpt_{args.dataset}_p2_fold{val_fold}"
    checkpoint_val = None
    checkpoint_test = None

    if init_from == "resume" and os.path.exists(
        os.path.join(checkpoint_out_dir, save_filename + ".pt")
    ):
        checkpoint = torch.load(
            os.path.join(checkpoint_out_dir, save_filename + ".pt"),
            map_location=device,
        )
        model.load_state_dict(checkpoint["model"])

        start_epoch = checkpoint["epoch"] + 1
        if start_epoch // 10 > 0:
            last_save_epoch = checkpoint["epoch"] // 10 * 10 - 1
            checkpoint_val = torch.load(
                os.path.join(
                    checkpoint_out_dir,
                    save_filename_val + f"_{last_save_epoch}.pt",
                ),
                map_location=device,
            )
            checkpoint_test = torch.load(
                os.path.join(
                    checkpoint_out_dir,
                    save_filename_test + f"_{last_save_epoch}.pt",
                ),
                map_location=device,
            )

        if start_epoch >= args.epochs:
            print(
                f"Training has already finished for {args.epochs} epochs, exiting..."
            )
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, final_filename))
            print(f"Saved final model to {os.path.join(checkpoint_out_dir, final_filename)}")
            return

        with open(log_txt, "w") as f:
            f.write(
                "resume from "
                + os.path.join(checkpoint_out_dir, save_filename + ".pt")
                + "\n"
            )
            f.write("start_epoch: " + str(start_epoch) + "\n")

        print(
            f"resume from {os.path.join(checkpoint_out_dir, save_filename + '.pt')}, "
            f"start_epoch {start_epoch}"
        )
    else:
        start_epoch = 0
        init_from = "scratch"

    model.to(device)

    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        print("prepare dataloader...")
        with open(log_txt, "w") as f:
            f.write("start_name: " + args.start_name + "\n")
            f.write(str(int(args.start_name)))
            f.write(str(val_sessions) + "\n")
            f.write(str(test_sessions) + "\n")
            f.write(str(train_sessions) + "\n")
            f.write("prepare dataloader...\n")

    # Dataset & metrics
    if args.dataset == "GPP":
        dataset_train, dataset_val, dataset_test = prepare_GPP_dataset(
            args.dataset_dir, train_sessions, val_sessions, test_sessions
        )

        val_domains = [
            int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 1 for s in val_sessions
        ]
        test_domains = [
            int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 1 for s in test_sessions
        ]
        globals()["target_domains"] = val_domains + test_domains

        with open(log_txt, "a") as f:
            f.write(f"len(train): {len(dataset_train)}\n")
            f.write(f"len(val): {len(dataset_val)}\n")
            f.write(f"len(test): {len(dataset_test)}\n")

        metrics = ["r2", "rmse", "pearsonr"]
        monitor = "r2"

    elif args.dataset == "MoBI":
        dataset_train, dataset_val, dataset_test = prepare_MoBI_dataset(
            args.dataset_dir, train_sessions, val_sessions, test_sessions
        )

        val_domains = [
            int(s.split("_")[0]) * 3 + int(s.split("_")[1]) - 4 for s in val_sessions
        ]
        test_domains = [
            int(s.split("_")[0]) * 3 + int(s.split("_")[1]) - 4 for s in test_sessions
        ]
        globals()["target_domains"] = val_domains + test_domains

        with open(log_txt, "a") as f:
            f.write(f"len(train): {len(dataset_train)}\n")
            f.write(f"len(val): {len(dataset_val)}\n")
            f.write(f"len(test): {len(dataset_test)}\n")

        metrics = ["r2", "rmse", "pearsonr"]
        monitor = "r2"

    elif args.dataset == "FBM":
        dataset_train, dataset_val, dataset_test = prepare_FBM_dataset(
            args.dataset_dir, train_sessions, val_sessions, test_sessions
        )
        print("args.dataset_dir", args.dataset_dir)

        val_domains = [int(s.split("_")[0]) - 2 for s in val_sessions]
        test_domains = [int(s.split("_")[0]) - 2 for s in test_sessions]
        globals()["target_domains"] = val_domains + test_domains

        print(f"len(train): {len(dataset_train)}")
        print(f"len(val): {len(dataset_val)}")
        print(f"len(test): {len(dataset_test)}")

        metrics = ["r2", "rmse", "pearsonr"]
        monitor = "r2"

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("finished dataset preparation")

    if init_from == "resume" and os.path.exists(
        os.path.join(checkpoint_out_dir, save_filename + ".pt")
    ):
        iter_num = checkpoint["epoch"] * (
            len(dataset_train) // args.batch_size // ddp_world_size
        )

    # Dataloaders
    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
        )

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=10,
            pin_memory=False,
            drop_last=True,
            shuffle=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )

    # GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # Optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        device_type,
    )

    if init_from == "resume":
        pretrained_ckpt = checkpoint
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free memory

    # Optional compile
    if args.compile:
        print("compiling the model... (takes ~1 minute)")
        with open(log_txt, "a") as f:
            f.write("compiling the model... (takes ~1 minute)\n")
        model = torch.compile(model)  # PyTorch 2.0+

    # Wrap in DDP
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # Logging / wandb
    if args.wandb_log and master_process:
        import wandb

        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_runname,
            dir=os.path.join(args.out_dir, "wandb"),
            resume=False,
        )

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // ddp_world_size
    lr_schedule_values = cosine_scheduler(
        args.learning_rate,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    print("num_training_steps_per_epoch", num_training_steps_per_epoch)

    # Training loop
    best_val_monitor_metric = -10
    best_test_monitor_metric = -10
    if init_from == "resume" and "best_val_monitor_metric" in pretrained_ckpt:
        best_val_monitor_metric = pretrained_ckpt["best_val_monitor_metric"]
        best_test_monitor_metric = pretrained_ckpt["best_test_monitor_metric"]
    best_test_metric = None

    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model

    for epoch in range(start_epoch, args.epochs):
        for step, batch in enumerate(data_loader_train):
            # Shape handling for EEG_X
            if model_name.split("-")[0] not in [
                "SPaRCNet",
                "ContraWR",
                "FFCL",
                "STtransformer",
                "EEGConformer",
                "TCN",
            ]:
                if isinstance(batch["EEG_X"], np.ndarray):
                    batch["EEG_X"] = np.expand_dims(batch["EEG_X"], axis=1)
                elif isinstance(batch["EEG_X"], torch.Tensor):
                    batch["EEG_X"] = batch["EEG_X"].unsqueeze(1)

            if model_name.split("-")[0] in ["TCN"]:
                if isinstance(batch["EEG_X"], np.ndarray):
                    batch["EEG_X"] = np.transpose(batch["EEG_X"], (0, 2, 1))
                elif isinstance(batch["EEG_X"], torch.Tensor):
                    batch["EEG_X"] = batch["EEG_X"].permute(0, 2, 1)

            data = {
                "EEG": batch["EEG_X"].float().to(device, non_blocking=True)
                if "EEG_X" in batch
                else None,
                "domain": batch["domain"].to(device, non_blocking=True)
                if "domain" in batch
                else None,
                "Y": batch["Y"][:, -1, :].to(device, non_blocking=True)
                if "Y" in batch
                else None,
            }

            # Learning rate schedule
            lr = (
                lr_schedule_values[iter_num]
                if args.decay_lr
                else args.learning_rate
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient sync strategy for DDP
            if ddp:
                model.require_backward_grad_sync = (
                    (step + 1) % args.gradient_accumulation_steps == 0
                )

            # Forward / backward
            with ctx:
                loss, log = model(data, metrics=metrics)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            if (iter_num + 1) % args.log_interval == 0 and master_process:
                message = f"Epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: "
                for key, value in log.items():
                    message += f"{key.split('/')[-1]} {value:.4f}, "
                print(message[:-2])
                with open(log_txt, "a") as f:
                    f.write(message[:-2] + "\n")

                if args.wandb_log:
                    import wandb

                    log_train = log.copy()
                    log_train.update({"iter": iter_num, "lr": lr})
                    wandb.log(log_train)

            t1 = time.time()
            _ = t1 - t0
            t0 = t1

            iter_num += 1
            local_iter_num += 1

        # Validation
        val_is_better = False
        test_is_better = False
        test_metric = []
        val_metric = []

        log_val = evaluate(model, data_loader_val, metrics)
        if master_process:
            print("=" * 10)
            with open(log_txt, "a") as f:
                f.write("=" * 10 + "\n")

            message = "Evaluate EEG: "
            for key, value in log_val.items():
                message += f"{key.split('/')[-1]} {value:.4f}, "
            print(message[:-2])
            print("=" * 10)

            with open(log_txt, "a") as f:
                f.write(message[:-2])
                f.write("=" * 10 + "\n")
                tmp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                f.write(f"end time: {tmp_time}\n")

            if args.wandb_log:
                import wandb

                wandb.log(log_val)

        # Test
        log = evaluate(model, data_loader_test, metrics)
        log_test = {k.replace("val", "test"): v for k, v in log.items()}
        if master_process:
            print("=" * 10)
            with open(log_txt, "a") as f:
                f.write("=" * 10 + "\n")

            message = "Test EEG: "
            for key, value in log_test.items():
                message += f"{key.split('/')[-1]} {value:.4f}, "
            print(message[:-2])
            print("=" * 10)

            with open(log_txt, "a") as f:
                f.write(message[:-2])
                f.write("=" * 10 + "\n")
                tmp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                f.write(f"end time: {tmp_time}\n")

            if args.wandb_log:
                import wandb

                wandb.log(log_test)

        monitor_metric_val = "val/" + monitor
        monitor_metric_test = "test/" + monitor

        if log_val[monitor_metric_val] > best_val_monitor_metric:
            val_is_better = True
            best_val_monitor_metric = log_val[monitor_metric_val]
            best_test_metric = log_test.copy()

        if log_test[monitor_metric_test] > best_test_monitor_metric:
            test_is_better = True
            best_test_monitor_metric = log_test[monitor_metric_test]
            best_val_metric = log_val.copy()

        if val_is_better:
            test_metric.append(log_test.copy())
        if test_is_better:
            val_metric.append(log_val.copy())

        if master_process:
            rvalue = -1.0
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "epoch": epoch,
                "args": vars(args),
                "best_val_monitor_metric": best_val_monitor_metric,
                "best_test_monitor_metric": best_test_monitor_metric,
            }

            print(f"saving checkpoint to {checkpoint_out_dir}")
            with open(log_txt, "a") as f:
                f.write(f"saving checkpoint to {checkpoint_out_dir}\n")

            torch.save(
                checkpoint, os.path.join(checkpoint_out_dir, save_filename + ".pt")
            )

            if val_is_better:
                checkpoint_test = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "epoch": epoch,
                    "args": vars(args),
                }
                for key, value in test_metric[-1].items():
                    checkpoint_test[key] = value
                    if "pearsonr" in key:
                        rvalue = value
                torch.save(
                    checkpoint_test,
                    os.path.join(
                        checkpoint_out_dir,
                        save_filename_test + f"_{rvalue:.4f}.pt",
                    ),
                )

            if test_is_better:
                checkpoint_val = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "epoch": epoch,
                    "args": vars(args),
                }
                for key, value in val_metric[-1].items():
                    checkpoint_val[key] = value
                    if "pearsonr" in key:
                        rvalue = value
                torch.save(
                    checkpoint_val,
                    os.path.join(
                        checkpoint_out_dir,
                        save_filename_val + f"_{rvalue:.4f}.pt",
                    ),
                )

            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                with open(log_txt, "a") as f:
                    f.write(
                        f"saving checkpoint {epoch} to {checkpoint_out_dir}\n"
                    )
                if checkpoint_val is not None:
                    torch.save(
                        checkpoint_val,
                        os.path.join(
                            checkpoint_out_dir,
                            save_filename_val + f"_{epoch}.pt",
                        ),
                    )
                if checkpoint_test is not None:
                    torch.save(
                        checkpoint_test,
                        os.path.join(
                            checkpoint_out_dir,
                            save_filename_test + f"_{epoch}.pt",
                        ),
                    )

    if master_process and best_test_metric is not None:
        test_metric = best_test_metric
        message = "Best test metrics :"
        for key, value in test_metric.items():
            message += f"{key.split('/')[-1]} {value:.5f},"
        print(message[:-2])
        with open(log_txt, "a") as f:
            f.write(message[:-2] + "\n")

    if ddp:
        destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser("VQ training script", add_help=False)
    parser.add_argument(
        "--out_dir",
        default="./results/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--dataset_dir",
        default="/data/FT_dataset/",
        help="path where data is",
    )
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--dataset", default="GPP", help="dataset: GPP / MoBI / FBM")

    parser.add_argument("--wandb_log", default=False, action="store_true")
    parser.add_argument("--wandb_project", default="GPP_SI")
    parser.add_argument("--wandb_runname", default="GPP_baseline")
    parser.add_argument("--wandb_api_key", type=str)

    # Training args
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=2, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--block_size", default=512, type=int)

    parser.add_argument("--test_fold", default=10, type=int)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        metavar="LR",
        help="learning rate (default: 2e-5)",
    )
    parser.add_argument("--min_lr", type=float, default=2e-6)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="weight decay (default: 0.05)",
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
        help="clip gradients at this value, or disable if == 0.0",
    )
    parser.add_argument(
        "--decay_lr",
        default=True,
        action="store_false",
        help="use cosine LR decay if True (default True, set flag to disable)",
    )
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--compile", default=False, action="store_true")
    parser.add_argument("--start_name", default="1", help="subject to start")
    parser.add_argument("--model_name", default="TSception", help="model to use")

    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--pretrained", default=False, action="store_true")
    parser.add_argument(
        "--pretrained_model_path",
        default="/root/SI_GPP/results/checkpoints_400/deepConvNet/ckpt_0.5682.pt",
        help="path to pretrained model (for deepConvNet variants)",
    )
    parser.add_argument(
        "--info",
        default="baseline_FBM",
        help="run info tag, will be suffixed with dataset name",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name
    start_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    log_txt = f"./results/{model_name}_{args.start_name}_{start_time}.txt"

    args.min_lr = args.learning_rate * 0.1

    # Folds & sessions per dataset
    if args.dataset == "GPP":
        test_fold = args.test_fold * 2 + 1
        val_fold = args.test_fold * 2
        val_idx = [i for i in range(5 * val_fold, 5 * (val_fold + 1))]
        test_idx = [i for i in range(5 * test_fold, 5 * (test_fold + 1))]
        sessions = ["1", "2"]
        sbjs = [i for i in range(50)]

    elif args.dataset == "FBM":
        test_fold = args.test_fold
        val_fold = (args.test_fold + 6) % 9 + 2
        val_idx = [f"0{args.test_fold}"[-2:]]
        test_idx = [f"0{args.test_fold}"[-2:]]
        sessions = ["01"]
        sbjs = ["02", "03", "04", "05", "06", "07", "08", "09", "10"]
        args.num_eeg_channels = 59
        args.motor_input_dim = 8

    # elif args.dataset == "MoBI":
    #     val_idx = [6]
    #     test_idx = [7]
    #     sessions = ["0", "1", "2"]
    #     sbjs = [i for i in range(8)]

    args.wandb_runname = (
        f"{model_name}_{args.info}_F{args.test_fold}_pretrained"
        if args.pretrained
        else f"{model_name}_{args.info}_F{args.test_fold}_nonpretrained"
    )

    if args.dataset not in args.info:
        args.info = args.info + "_" + args.dataset

    args.ch_names = [
        "FP1",
        "FZ",
        "F3",
        "F7",
        "FC5",
        "FC1",
        "C3",
        "T7",
        "CP5",
        "CP1",
        "PZ",
        "P3",
        "P7",
        "O1",
        "OZ",
        "O2",
        "P4",
        "P8",
        "CP6",
        "CP2",
        "CZ",
        "C4",
        "T8",
        "FC6",
        "FC2",
        "F4",
        "F8",
        "FP2",
        "AF7",
        "AF3",
        "AFZ",
        "F1",
        "F5",
        "FT7",
        "FC3",
        "C1",
        "C5",
        "TP7",
        "CP3",
        "P1",
        "P5",
        "PO7",
        "PO3",
        "POZ",
        "PO4",
        "PO8",
        "P6",
        "P2",
        "CPZ",
        "CP4",
        "TP8",
        "C6",
        "C2",
        "FC4",
        "FT8",
        "F6",
        "AF8",
        "AF4",
        "F2",
    ]

    train_sessions = []
    val_sessions = []
    test_sessions = []

    for idx in val_idx:
        for session in sessions:
            val_sessions.append(f"{idx}_{session}")
    for idx in test_idx:
        for session in sessions:
            test_sessions.append(f"{idx}_{session}")
    for j in sbjs:
        if j not in val_idx and j not in test_idx:
            for session in sessions:
                train_sessions.append(f"{j}_{session}")

    # Build adjacency matrix
    ch_names = [
        "Fp1",
        "Fz",
        "F3",
        "F7",
        "FC5",
        "FC1",
        "C3",
        "T7",
        "CP5",
        "CP1",
        "Pz",
        "P3",
        "P7",
        "O1",
        "Oz",
        "O2",
        "P4",
        "P8",
        "CP6",
        "CP2",
        "Cz",
        "C4",
        "T8",
        "FC6",
        "FC2",
        "F4",
        "F8",
        "Fp2",
        "AF7",
        "AF3",
        "AFz",
        "F1",
        "F5",
        "FT7",
        "FC3",
        "C1",
        "C5",
        "TP7",
        "CP3",
        "P1",
        "P5",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "P6",
        "P2",
        "CPz",
        "CP4",
        "TP8",
        "C6",
        "C2",
        "FC4",
        "FT8",
        "F6",
        "AF8",
        "AF4",
        "F2",
    ]

    channel_neighbors = {
        "Fp1": ["AF7", "AF3", "AFz"],
        "Fp2": ["AF4", "AF8", "AFz"],
        "AF7": ["Fp1", "AF3", "F7"],
        "AF3": ["Fp1", "F3", "AF7", "AFz"],
        "AFz": ["Fp1", "Fp2", "Fz", "AF3", "AF4"],
        "AF4": ["Fp2", "F4", "AFz", "AF8"],
        "AF8": ["Fp2", "AF4", "F8"],
        "F7": ["AF7", "FT7", "F5"],
        "F5": ["AF7", "AF3", "FC5", "F7", "F3"],
        "F3": ["AF3", "FC3", "F5", "F1"],
        "F1": ["AF3", "AFz", "FC1", "F3", "Fz"],
        "Fz": ["AFz", "F1", "F2"],
        "F2": ["AFz", "AF4", "FC2", "Fz", "F4"],
        "F4": ["AF4", "FC4", "F2", "F6"],
        "F6": ["AF4", "AF8", "FC6", "F4", "F8"],
        "F8": ["AF8", "FT8", "F6"],
        "FT7": ["F7", "T7", "FC5"],
        "FC5": ["F5", "C5", "FT7", "FC3"],
        "FC3": ["F3", "C3", "FC5", "FC1"],
        "FC1": ["F1", "C1", "FC3"],
        "FC2": ["F2", "C2", "FC4"],
        "FC4": ["F4", "C4", "FC2", "FC6"],
        "FC6": ["F6", "C6", "FC4", "FT8"],
        "FT8": ["F8", "T8", "FC6"],
        "T7": ["FT7", "TP7", "C5"],
        "C5": ["FC5", "CP5", "T7", "C3"],
        "C3": ["FC3", "CP3", "C5", "C1"],
        "C1": ["FC1", "CP1", "C3", "Cz"],
        "Cz": ["CPz", "C1", "C2"],
        "C2": ["FC2", "CP2", "Cz", "C4"],
        "C4": ["FC4", "CP4", "C2", "CP6"],
        "C6": ["FC6", "CP6", "C4", "T8"],
        "T8": ["FT8", "TP8", "C6"],
        "TP7": ["T7", "P7", "CP5"],
        "CP5": ["C5", "P5", "TP7", "CP3"],
        "CP3": ["C3", "P3", "CP5", "CP1"],
        "CP1": ["C1", "P1", "CP3", "CPz"],
        "CPz": ["Cz", "Pz", "CP1", "CP2"],
        "CP2": ["C2", "P2", "CPz", "CP4"],
        "CP4": ["C4", "P4", "CP2", "CP6"],
        "CP6": ["C6", "P6", "TP8"],
        "TP8": ["T8", "P8", "CP6"],
        "P7": ["TP7", "PO7", "P5"],
        "P5": ["CP5", "PO7", "PO3", "P7", "P3"],
        "P3": ["CP3", "PO3", "P5", "P1"],
        "P1": ["CP1", "PO3", "POz", "P3", "Pz"],
        "Pz": ["CPz", "POz", "P1", "P2"],
        "P2": ["CP2", "POz", "PO4", "Pz", "P4"],
        "P4": ["CP4", "PO4", "P2", "P6"],
        "P6": ["CP6", "PO4", "PO8", "P4", "P8"],
        "P8": ["TP8", "PO8", "P6"],
        "PO7": ["P7", "O1", "PO3"],
        "PO3": ["P3", "O1", "Oz", "PO7", "POz"],
        "POz": ["Pz", "Oz", "PO3", "PO4"],
        "PO4": ["P4", "Oz", "O2", "POz", "PO8"],
        "PO8": ["P8", "O2", "PO4"],
        "O1": ["PO7", "PO3", "Oz"],
        "Oz": ["POz", "O1", "O2"],
        "O2": ["PO8", "PO4", "Oz"],
    }

    num_channels = len(ch_names)
    adj_matrix = np.zeros((num_channels, num_channels), dtype=np.float32)

    for channel, neighbors in channel_neighbors.items():
        i = ch_names.index(channel)
        for neighbor in neighbors:
            j = ch_names.index(neighbor)
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0

    np.fill_diagonal(adj_matrix, 1.0)
    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to("cuda")

    config = load_config("./config.yaml")
    config.eegnet.adj_matrix = adj_matrix_tensor

    if args.dataset == "GPP":
        config.eegnet.num_chan_eeg = 59
        config.num_chan_eeg = 59
        config.eegnet.num_chan_kin = 6
        config.num_chan_kin = 6

    elif args.dataset == "FBM":
        config.eegnet.num_chan_eeg = 59
        config.num_chan_eeg = 59
        config.eegnet.num_chan_kin = 8
        config.num_chan_kin = 8

    config.eegnet.eeg.time_step = 400
    config.eegnet.loss_name = "mse"

    # Per-GPU batch size
    args.batch_size = args.batch_size // torch.cuda.device_count()
    args.min_lr = args.learning_rate * 0.1
    args.resume = True

    print("train_sessions:", train_sessions)
    print("val_sessions:", val_sessions)
    print("test_sessions:", test_sessions)

    # Run training
    main(args)
