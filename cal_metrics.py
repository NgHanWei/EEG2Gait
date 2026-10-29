"""
This training/evaluation script can be run both on a single GPU in debug mode,
and also in a larger run with Distributed Data Parallel (DDP).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 GPUs on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 GPUs across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py

(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import (
    prepare_GPP_dataset,
    prepare_ss_dataset,
    prepare_MoBI_ss_dataset,
    get_metrics,
    get_metrics_single,
)
from utils.myaml import load_config
import argparse
import itertools

from models import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,  # kept for potential future use, though not used below
    FFCL,
    STTransformer,
)

master_process = None
device = None
dtype = None
ctx = None
ddp_rank = None
device_type = None
ddp = None
ddp_world_size = None
ddp_local_rank = None


def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    backend = "nccl"
    device = "cuda"
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

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

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


@torch.no_grad()
def evaluate(model, dataloader, metrics):
    global ctx, model_name
    model.eval()
    pred = []
    true = []
    total_loss = []
    log = {}

    for _, batch in enumerate(dataloader):
        if model_name.split("-")[0] not in ["SPaRCNet", "ContraWR", "FFCL", "STtransformer", "EEGConformer", "TCN"]:
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
            "EEG": batch["EEG_X"].float().to(device, non_blocking=True) if "EEG_X" in batch else None,
            "domain": batch["domain"].to(device, non_blocking=True) if "domain" in batch else None,
            "Y": batch["Y"].to(device, non_blocking=True) if "Y" in batch else None,
            "EEG_input_chans": batch["EEG_input_chans"].to(device, non_blocking=True)
            if "EEG_input_chans" in batch
            else None,
            "EEG_input_time": batch["EEG_input_time"].to(device, non_blocking=True)
            if "EEG_input_time" in batch
            else None,
        }

        with ctx:
            loss, logits = model(data, metrics=metrics)

        pred.append(logits.cpu())
        true.append(data["Y"].cpu())
        total_loss.append(loss)

    pred = torch.cat(pred, dim=0)
    true = torch.cat(true, dim=0)
    total_loss = np.mean(total_loss)

    if "r2" in metrics:
        results = get_metrics_single(pred.numpy(), true.numpy(), metrics, is_binary=False)
    elif "f1_weighted" not in metrics:
        results = get_metrics(torch.sigmoid(pred).numpy(), true, metrics, is_binary=True)
    else:
        results = get_metrics(pred.numpy(), true.numpy(), metrics, is_binary=False)

    log["val/total_loss"] = total_loss
    for key, value in results.items():
        log[f"val/{key}"] = value

    model.train()
    return log


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    if args.dataset == "MoBI":
        session_start = 0
        session_end = 3
        sbj_start = 0
        sbj_end = 8
        args.dataset_dir = "/scratch/users/ntu/fuxi0010/MoBI_npy_ss"
        args.learning_rate = 5e-3

    elif args.dataset == "GPP":
        session_start = 2
        session_end = 3
        sbj_start = 36
        sbj_end = 37
        args.dataset_dir = "/scratch/users/ntu/fuxi0010/GPP/trialwise_dataset/normal/"

    result_single = {}

    for i in range(int(sbj_start), int(sbj_end)):
        for j in range(int(session_start), int(session_end)):
            args.session = f"{i}_{j}"
            args.wandb_runname = f"sbj_dependent_{model_name}_{args.session}"

            checkpoint_out_dir = os.path.join(args.out_dir, args.dataset, f"{model_name}_{args.loss_name}")
            checkpoint_path = None
            if os.path.isdir(checkpoint_out_dir):
                for file in os.listdir(checkpoint_out_dir):
                    if file.endswith(".pt"):
                        sbj_name = file.split("_")[2]
                        session_name = file.split("_")[3]
                        if sbj_name == str(i) and session_name == str(j):
                            checkpoint_path = os.path.join(checkpoint_out_dir, file)
                            break

            if checkpoint_path is None:
                print(f"Checkpoint not found for subject {i}, session {j}")
                continue

            if master_process:
                os.makedirs(checkpoint_out_dir, exist_ok=True)
                print("Preparing dataloader...")
                with open(log_txt, "a") as f:
                    f.write("session: " + args.session + "\n")
                    f.write("prepare dataloader...\n")

            if args.dataset == "GPP":
                dataset_train, dataset_val, dataset_test = prepare_ss_dataset(
                    args.dataset_dir, args.session, 100, 5, 15
                )
                if master_process:
                    with open(log_txt, "a") as f:
                        f.write(
                            "dataset_train, dataset_val, dataset_test: "
                            f"{len(dataset_train)} {len(dataset_val)} {len(dataset_test)}\n"
                        )
                    print("dataset_dir:", args.dataset_dir)
                    print(
                        "len(dataset_train), len(dataset_val), len(dataset_test)",
                        len(dataset_train),
                        len(dataset_val),
                        len(dataset_test),
                    )
                n_classes = 6
                regression = True
                metrics = ["r2", "rmse", "pearsonr", "mae"]
                monitor = "r2"
                test_conditions = []
                for test_condition in itertools.product([True, False], repeat=3):
                    if sum(test_condition) == 0:
                        continue
                    test_conditions.append(list(test_condition).insert(2, False))

            if args.dataset == "MoBI":
                dataset_train, dataset_val, dataset_test = prepare_MoBI_ss_dataset(args.dataset_dir, args.session)
                if master_process:
                    with open(log_txt, "a") as f:
                        f.write(
                            "dataset_train, dataset_val, dataset_test: "
                            f"{len(dataset_train)} {len(dataset_val)} {len(dataset_test)}\n"
                        )
                    print("dataset_dir:", args.dataset_dir)
                    print(
                        "len(dataset_train), len(dataset_val), len(dataset_test)",
                        len(dataset_train),
                        len(dataset_val),
                        len(dataset_test),
                    )
                n_classes = 6
                regression = True
                metrics = ["r2", "rmse", "pearsonr", "mae"]
                monitor = "r2"
                test_conditions = []
                for test_condition in itertools.product([True, False], repeat=3):
                    if sum(test_condition) == 0:
                        continue
                    test_conditions.append(list(test_condition).insert(2, False))

            print("Finished dataset preparation.")

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
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
                )
                data_loader_val = torch.utils.data.DataLoader(
                    dataset_val,
                    sampler=sampler_val,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=10,
                    pin_memory=True,
                    drop_last=False,
                )
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
                )
                data_loader_test = torch.utils.data.DataLoader(
                    dataset_test,
                    sampler=sampler_test,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=10,
                    pin_memory=True,
                    drop_last=False,
                )
            else:
                data_loader_train = torch.utils.data.DataLoader(
                    dataset_train,
                    batch_size=args.batch_size,
                    num_workers=10,
                    pin_memory=True,
                    drop_last=True,
                    shuffle=True,
                )
                data_loader_val = torch.utils.data.DataLoader(
                    dataset_val,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=10,
                    pin_memory=True,
                    drop_last=False,
                    shuffle=False,
                )
                data_loader_test = torch.utils.data.DataLoader(
                    dataset_test,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=10,
                    pin_memory=True,
                    drop_last=False,
                    shuffle=False,
                )

            if model_name.startswith("SPaRCNet"):
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

            if model_name.startswith("STtransformer"):
                model = STTransformer(
                    n_classes=config.eegnet.num_chan_kin,
                    channel_legnth=config.eegnet.eeg.time_step,
                    n_channels=config.eegnet.num_chan_eeg,
                ).to(device)

            if model_name.startswith("EEGNet"):
                from EEGNet import EEGNet

                model = EEGNet(config).to(device)

            if model_name.startswith("deepConvNet"):
                from models.DCN_model import deepConvNet

                model = deepConvNet(config).to(device)

            if model_name.startswith("TCN"):
                from models.tcn_model import TCN

                model = TCN(config).to(device)

            if model_name.startswith("GaitGraph"):
                from models.GaitGraph_model import GaitGraph_tmp

                model = GaitGraph_tmp(config).to(device)

            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model"])

            if master_process:
                with open(log_txt, "a") as f:
                    f.write("\n\n\nload checkpoint from " + checkpoint_path + "\n")
                    for key in ckpt.keys():
                        if key in ["model", "optimizer"]:
                            continue
                        f.write(key + " " + str(ckpt[key]) + "\n")

            model.to(device)

            if args.compile:
                if master_process:
                    print("Compiling the model (PyTorch 2.x)...")
                    with open(log_txt, "a") as f:
                        f.write("compiling the model... (takes a while)\n")
                model = torch.compile(model)

            if ddp:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

            if args.wandb_log and master_process:
                import wandb

                os.environ["WANDB_API_KEY"] = args.wandb_api_key
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_runname,
                    dir=os.path.join(args.out_dir, "wandb"),
                    resume=False,
                )

            is_better = True
            test_metric_list = []
            best_test_metric = None

            log = evaluate(model, data_loader_test, metrics)
            log_test = {k.replace("val", "test"): v for k, v in log.items()}

            if master_process:
                print("=" * 10)
                with open(log_txt, "a") as f:
                    f.write("=" * 10 + "\n")
                message = "Test EEG: "
                for key, value in log_test.items():
                    name = key.split("/")[-1]
                    message += f"{name} {value:.4f}, "
                    if name not in result_single:
                        result_single[name] = []
                    result_single[name].append(value)
                print(message[:-2])
                print("=" * 10)
                with open(log_txt, "a") as f:
                    f.write(message[:-2])
                    f.write("=" * 10 + "\n")
                    tmp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    f.write(f"end time: {tmp_time}\n")
                if args.wandb_log:
                    wandb.log(log_test)

            if is_better:
                test_metric_list.append(log_test.copy())
                best_test_metric = test_metric_list.copy()

            if master_process and best_test_metric is not None:
                for tm in best_test_metric:
                    for key, _ in tm.items():
                        m = key.split("/")[0][5:]
                        break
                    message = f"Best test metrics {m}:"
                    for key, value in tm.items():
                        message += f" {key.split('/')[-1]} {value:.5f},"
                    print(message[:-2])
                    with open(log_txt, "a") as f:
                        f.write(message[:-2] + "\n")

    if master_process:
        print("result_single:", result_single)
        csv_path = "./results/result_single.csv"
        with open(csv_path, "w") as f:
            f.write("metric,")
            for i in range(int(sbj_start), int(sbj_end)):
                for j in range(int(session_start), int(session_end)):
                    f.write(f"sbj_{i}_session_{j},")
            f.write("\n")
            for metric, values in result_single.items():
                f.write(metric + ",")
                for value in values:
                    f.write(f"{value:.4f},")
                f.write("\n")

    if ddp:
        destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser("VQ training script", add_help=False)
    parser.add_argument(
        "--out_dir",
        default="./final_results/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--dataset_dir",
        default="/scratch/users/ntu/fuxi0010/GPP/trialwise_dataset/normal/",
        help="path where data is",
    )
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--dataset", default="GPP", help="dataset: GPP")
    parser.add_argument(
        "--loss_name",
        default="mse",
        help="loss name: mse, freq_mse, reward_mse, freq_reward_mse",
    )

    parser.add_argument("--wandb_log", default=False, action="store_true")
    parser.add_argument("--wandb_project", default="GPP")
    parser.add_argument("--wandb_runname", default="GPP")
    parser.add_argument("--wandb_api_key", type=str)

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=2, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--block_size", default=512, type=int)

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        metavar="LR",
    )
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.0,
        help="clip gradients at this value, or disable if == 0.0",
    )
    parser.add_argument("--decay_lr", default=True, action="store_false")
    parser.add_argument("--seed", default=1337, type=int)

    parser.add_argument("--compile", default=False, action="store_true")
    parser.add_argument("--start_name", default="1", help="subject to start")
    parser.add_argument("--model_name", default="GaitGraph", help="model to use")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name
    args.dataset = "GPP"
    start_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    log_txt = f"./results/{model_name}_{args.start_name}_{start_time}.txt"

    config = load_config("./config.yaml")
    config.eegnet.eeg.time_step = 200
    config.eegnet.loss_name = args.loss_name

    with open(log_txt, "a") as f:
        f.write(f"Available GPUs on this node: {torch.cuda.device_count()}\n")

    args.batch_size = 512 // torch.cuda.device_count()
    main(args)
