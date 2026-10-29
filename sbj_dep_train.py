"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
from contextlib import nullcontext
import torch
import torch._dynamo.config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from utils.utils import cosine_scheduler, prepare_GPP_dataset, prepare_ss_dataset, prepare_MoBI_ss_dataset, get_metrics
import argparse
import numpy as np
from utils.myaml import load_config
import itertools
import braindecode.models
from collections import OrderedDict

# from models import (
#     SPaRCNet,
#     ContraWR,
#     CNNTransformer,
#     FFCL,
#     STTransformer,
# )
from models.sparcnet import SPaRCNet
from models.contrawr import ContraWR
from models.ffcl import FFCL
from models.st_transformer import STTransformer
import time
from torch.utils.data import Dataset, Subset

master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None

def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        #assert gradient_accumulation_steps % ddp_world_size == 0
        #gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=ptdtype)

def get_exist_sess(checkpoint_out_dir):
    exist_sess = []
    for file in os.listdir(checkpoint_out_dir):
        if file.endswith('.pt'):
            exist_sess.append(file.split('_')[2] + '_' + file.split('_')[3])
    return exist_sess

@torch.no_grad
def evaluate(model, dataloader, metrics):
    global ctx, model_name
    model.eval()
    pred = []
    true = []
    total_loss = []
    log = {}
    for _, (batch) in enumerate(dataloader):
        if model_name.split('-')[0] not in ['SPaRCNet', 'ContraWR', 'FFCL', 'STtransformer', 'EEGConformer', 'TCN']:
            if isinstance(batch["EEG_X"], np.ndarray):
                batch["EEG_X"] = np.expand_dims(batch["EEG_X"], axis=1)  # NumPy
            elif isinstance(batch["EEG_X"], torch.Tensor):
                batch["EEG_X"] = batch["EEG_X"].unsqueeze(1)  # PyTorch

        if model_name.split('-')[0] in ['TCN']:
            if isinstance(batch["EEG_X"], np.ndarray):
                batch["EEG_X"] = np.transpose(batch["EEG_X"], (0, 2, 1))  # NumPy
            elif isinstance(batch["EEG_X"], torch.Tensor):
                batch["EEG_X"] = batch["EEG_X"].permute(0, 2, 1)  # PyTorch (np.transpose)

                    
        data = {
            'EEG': batch['EEG_X'].float().to(device, non_blocking=True) if 'EEG_X' in batch else None,
            'domain': batch['domain'].to(device, non_blocking=True) if 'domain' in batch else None,
            'Y': batch['Y'].to(device, non_blocking=True) if 'Y' in batch else None,
            'EEG_input_chans': batch['EEG_input_chans'].to(device, non_blocking=True) if 'EEG_input_chans' in batch else None,
            'EEG_input_time': batch['EEG_input_time'].to(device, non_blocking=True) if 'EEG_input_time' in batch else None,
        }

        with ctx:
            loss, logits = model(data, metrics=metrics)
        
        # print(data['Y'].shape)
        pred.append(logits.cpu())
        true.append(data['Y'].cpu())
        total_loss.append(loss)

    predictions = torch.cat(true, dim=0).numpy()

    print("Predictions shape:", predictions.shape)

    # Save for predictions
    np.save("predictions.npy", predictions)

    print("Saved predictions.npy")

    pred = torch.cat(pred, dim=0)
    true = torch.cat(true, dim=0)
    total_loss = np.mean(total_loss)

    if 'r2' in metrics:
        results = get_metrics(pred.numpy(), true.numpy(), metrics, is_binary=False)
    elif 'f1_weighted' not in metrics:
        # binary classification
        results = get_metrics(torch.sigmoid(pred).numpy(), true, metrics, is_binary=True)
    else:
        # multi-class classification
        results = get_metrics(pred.numpy(), true.numpy(), metrics, is_binary=False)
    
    log[f'val/total_loss'] = total_loss
    for key, value in results.items():
        log[f'val/{key}'] = value

    model.train()
    
    return log


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)
    if args.dataset == 'MoBI':
        session_start = 0
        session_end = 3
        sbj_start = 0
        sbj_end = 8
        args.dataset_dir = r'X:\xxxx\Data\MoBI\MoBI_npy_ss'
        args.learning_rate = 5e-3

    elif args.dataset == 'GPP':
        session_start = 1
        session_end = 3
        sbj_start = 25
        sbj_end = 26
        args.learning_rate = 0.5e-3
        ### Change Dataset Directory here.
        args.dataset_dir = r'/home/hanwei/Gait_ME/normal'

    checkpoint_out_dir = os.path.join(args.out_dir, args.dataset, model_name+f"_{args.loss_name}")
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
    
    for i in range(int(sbj_start), int(sbj_end)):
        for j in range(int(session_start), int(session_end)):
            args.session = str(i) + '_' + str(j)
            exist_sess = get_exist_sess(checkpoint_out_dir)
            # if args.session in exist_sess and model_name not in ['GaitGraph', 'GaitCNN']:
            #     continue
            
            args.wandb_runname = f'sbj_dependent_{model_name}_{args.session}'
            if master_process:
                
                print('prepare dataloader...')
                with open(log_txt, 'a') as f:
                    f.write('session: ' + args.session + '\n')
                    f.write('prepare dataloader...\n')

            if args.dataset == 'GPP':
                print(args.dataset_dir)
                print(args.session)
                dataset_train, dataset_val, dataset_test = prepare_ss_dataset(args.dataset_dir, args.session, 100, 5, 15)
                with open(log_txt, 'a') as f:
                    f.write('dataset_train, dataset_val, dataset_test: ' + str(len(dataset_train)) + ' ' + str(len(dataset_val)) + ' ' + str(len(dataset_test)) + '\n')
                print('dataset_dir:', args.dataset_dir)
                print('len(dataset_train), len(dataset_val), len(dataset_test)', len(dataset_train), len(dataset_val), len(dataset_test))
                n_classes = 6
                regression = True
                metrics = ["r2", "rmse", "pearsonr", "mae"]
                monitor = "r2"
                test_conditions = []
                for test_condition in list(itertools.product([True, False], repeat=3)):
                    if sum(test_condition) == 0:
                        continue
                    test_conditions.append(list(test_condition).insert(2, False))
            if args.dataset == 'MoBI':
                dataset_train, dataset_val, dataset_test = prepare_MoBI_ss_dataset(args.dataset_dir, args.session)
                with open(log_txt, 'a') as f:
                    f.write('dataset_train, dataset_val, dataset_test: ' + str(len(dataset_train)) + ' ' + str(len(dataset_val)) + ' ' + str(len(dataset_test)) + '\n')
                print('dataset_dir:', args.dataset_dir)
                print('len(dataset_train), len(dataset_val), len(dataset_test)', len(dataset_train), len(dataset_val), len(dataset_test))
                n_classes = 6
                regression = True
                metrics = ["r2", "rmse", "pearsonr", "mae"]
                monitor = "r2"
                test_conditions = []
                for test_condition in list(itertools.product([True, False], repeat=3)):
                    if sum(test_condition) == 0:
                        continue
                    test_conditions.append(list(test_condition).insert(2, False))
            print('finished!')
            
            

            if ddp:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
                )
                data_loader_train = torch.utils.data.DataLoader(
                    dataset_train, sampler=sampler_train,
                    batch_size=args.batch_size,
                    num_workers=10,
                    pin_memory=True,
                    drop_last=True,
                )
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
                )
                data_loader_val = torch.utils.data.DataLoader(
                    dataset_val, sampler=sampler_val,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=10,
                    pin_memory=True,
                    drop_last=False,
                )
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
                )
                data_loader_test = torch.utils.data.DataLoader(
                    dataset_test, sampler=sampler_test,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=10,
                    pin_memory=True,
                    drop_last=False,
                )
            else:
                data_loader_train = torch.utils.data.DataLoader(
                    dataset_train,
                    batch_size=args.batch_size,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True,
                    shuffle=False
                )
                data_loader_val = torch.utils.data.DataLoader(
                    dataset_val,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                    shuffle=False
                )
                data_loader_test = torch.utils.data.DataLoader(
                    dataset_test,
                    batch_size=int(1.5 * args.batch_size),
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                    shuffle=False
                )

            # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
            if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt_-1.0000.pt')):
                init_from = 'resume'
            else:
                init_from = 'scratch'
            init_from = 'scratch'
            with open(log_txt, 'a') as f:
                f.write('init_from: ' + init_from + '\n')
            iter_num = 0
            if model_name.startswith('EEGConformer'):
                from models.EEGConformer import MyEEGConformer
                
                model = MyEEGConformer(
                        n_outputs=config.eegnet.num_chan_kin,
                        n_chans=config.eegnet.num_chan_eeg,
                        n_times=config.eegnet.eeg.time_step,
                        final_fc_length=280,
                        add_log_softmax=False, 
                        loss_name=args.loss_name
                    ).to(device)
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
                    loss_name=args.loss_name
                ).to(device)

            elif model_name.startswith("ContraWR"):
                model = ContraWR(in_channels=config.eegnet.num_chan_eeg, n_classes=config.eegnet.num_chan_kin, fft=20,
                                steps=2, loss_name=args.loss_name).to(device)


            elif model_name.startswith("FFCL"):
                model = FFCL(
                    in_channels=config.eegnet.num_chan_eeg,
                    n_classes=config.eegnet.num_chan_kin,
                    fft=20,
                    steps=2,
                    sample_length=config.eegnet.eeg.time_step,
                    shrink_steps=20,
                    loss_name=args.loss_name
                ).to(device)

            elif model_name.startswith('TSception'):
                from models.TSception_model import TSception
                model = TSception(config).to(device)
                
            if model_name.startswith("STtransformer"):
                model = STTransformer(n_classes=config.eegnet.num_chan_kin,
                                    channel_legnth=config.eegnet.eeg.time_step,
                                    n_channels=config.eegnet.num_chan_eeg,
                                    loss_name=args.loss_name).to(device)
            if model_name.startswith('EEGNet'):
                from EEGNet import EEGNet
                model = EEGNet(config).to(device)
            if model_name.startswith('deepConvNet'):
                from models.DCN_model import deepConvNet
                model = deepConvNet(config).to(device)

            if model_name.startswith('TCN'):
                from models.tcn_model import TCN
                model = TCN(config).to(device)
            if model_name.startswith('GaitGraph'):
                from models.GaitGraph_model import GaitGraph_tmp
                model = GaitGraph_tmp(config, init_adj=False).to(device)
            if model_name.startswith('GaitCNN'):
                from models.GaitGraph_model import GaitCNN_tmp
                model = GaitCNN_tmp(config).to(device)
            if init_from == 'resume':
                checkpoint = torch.load(os.path.join(checkpoint_out_dir, 'ckpt_-1.0000.pt'), map_location=device)
                model.load_state_dict(checkpoint['model'])
                start_epoch = checkpoint['epoch'] + 1
                with open(log_txt, 'a') as f:
                    f.write('resume from ' + os.path.join(checkpoint_out_dir, 'ckpt_-1.0000.pt') + '\n')
                    f.write('start_epoch: ' + str(start_epoch) + '\n')
            elif init_from == 'scratch':
                start_epoch = 0

            model.to(device)

            # initialize a GradScaler. If enabled=False scaler is a no-op
            # scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

            # optimizer
            optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
            if init_from == 'resume':
                optimizer.load_state_dict(checkpoint['optimizer'])
            checkpoint = None # free up memory

            # compile the model
            if args.compile:
                print("compiling the model... (takes a ~minute)")
                with open(log_txt, 'a') as f:
                    f.write("compiling the model... (takes a ~minute)\n")
                unoptimized_model = model
                model = torch.compile(model) # requires PyTorch 2.0

            # wrap model into DDP container
            if ddp:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

            # logging
            if args.wandb_log and master_process:
                import wandb
                os.environ["WANDB_API_KEY"] = args.wandb_api_key
                wandb.init(project=args.wandb_project, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'), resume=False)

            num_training_steps_per_epoch = len(dataset_train) // args.batch_size // ddp_world_size
            lr_schedule_values = cosine_scheduler(
                args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
                warmup_epochs=args.warmup_epochs
            )


            # training loop
            best_val_metric = -10
            t0 = time.time()
            local_iter_num = 0 # number of iterations in the lifetime of this process
            raw_model = model.module if ddp else model # unwrap DDP container if needed
            print("Number of Epochs: ", args.epochs)
            for epoch in range(start_epoch, args.epochs):
                for step, (batch) in enumerate(data_loader_train):

                    # Assume that `batch["EEG_X"]` may be either a NumPy array or a PyTorch Tensor.
                    if model_name.split('-')[0] not in ['SPaRCNet', 'ContraWR', 'FFCL', 'STtransformer', 'EEGConformer', 'TCN']:
                        if isinstance(batch["EEG_X"], np.ndarray):
                            batch["EEG_X"] = np.expand_dims(batch["EEG_X"], axis=1)  # NumPy
                        elif isinstance(batch["EEG_X"], torch.Tensor):
                            batch["EEG_X"] = batch["EEG_X"].unsqueeze(1)  # PyTorch

                    if model_name.split('-')[0] in ['TCN']:
                        if isinstance(batch["EEG_X"], np.ndarray):
                            batch["EEG_X"] = np.transpose(batch["EEG_X"], (0, 2, 1))  # NumPy
                        elif isinstance(batch["EEG_X"], torch.Tensor):
                            batch["EEG_X"] = batch["EEG_X"].permute(0, 2, 1)  # PyTorch (np.transpose)

                                
                    data = {
                        'EEG': batch['EEG_X'].float().to(device, non_blocking=True) if 'EEG_X' in batch else None, #(1, 59, 200)
                        'domain': batch['domain'].to(device, non_blocking=True) if 'domain' in batch else None,
                        'Y': batch['Y'].to(device, non_blocking=True) if 'Y' in batch else None,
                        'EEG_input_chans': batch['EEG_input_chans'].to(device, non_blocking=True) if 'EEG_input_chans' in batch else None,
                        'EEG_input_time': batch['EEG_input_time'].to(device, non_blocking=True) if 'EEG_input_time' in batch else None,
                    }
                    

                    # determine and set the learning rate for this iteration
                    lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # forward backward update, with optional gradient accumulation to simulate larger batch size
                    # and using the GradScaler if data type is float16
                    if ddp:
                        # in DDP training we only need to sync gradients at the last micro step.
                        # the official way to do this is with model.no_sync() context manager, but
                        # I really dislike that this bloats the code and forces us to repeat code
                        # looking at the source of that context manager, it just toggles this variable
                        model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0
                    
                    with ctx:
                        loss, log = model(data, metrics=metrics)
                        loss = loss / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    # backward pass, with gradient scaling if training in fp16
                    scaler.scale(loss).backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # clip the gradient
                        if args.grad_clip != 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        # step the optimizer and scaler if training in fp16
                        scaler.step(optimizer)
                        scaler.update()
                        # flush the gradients as soon as we can, no need for this memory anymore
                        optimizer.zero_grad(set_to_none=True)

                    # evaluate the loss on train/val sets and write checkpoints
                    if (iter_num + 1) % args.log_interval == 0 and master_process:
                        message = f'Epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: '
                        for key, value in log.items():
                            message += key.split('/')[-1]
                            message += f' {value:.4f}, '
                        print(message[:-2])
                        with open(log_txt, 'a') as f:
                            f.write(message[:-2] + '\n')
                        
                        if args.wandb_log:
                            log_train = log.copy()
                            log_train.update({
                                "iter": iter_num,
                                "lr": lr
                            })
                            wandb.log(log_train)

                    # timing and logging
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1

                    iter_num += 1
                    local_iter_num += 1

                is_better = False
                test_metric = []
            
                # validation
                log_val = evaluate(model, data_loader_val, metrics)
                if master_process:
                    print('='* 10)
                    with open(log_txt, 'a') as f:
                        f.write('='* 10 + '\n')
                    message = f'Evaluate EEG: '
                    for key, value in log_val.items():
                        message += key.split('/')[-1]
                        message += f' {value:.4f}, '
                    print(message[:-2])
                    print('='* 10)
                    with open(log_txt, 'a') as f:
                        f.write(message[:-2])
                        f.write('='* 10 + '\n')
                        tmp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        f.write(f'end time: {tmp_time}\n')
                    if args.wandb_log:
                        wandb.log(log_val)

                # test
                log = evaluate(model, data_loader_test, metrics)
                log_test = {k.replace('val', 'test'): v for k, v in log.items()}
                if master_process:
                    print('='* 10)
                    with open (log_txt, 'a') as f:
                        f.write('='* 10 + '\n')
                    message = f'Test EEG: '
                    for key, value in log_test.items():
                        message += key.split('/')[-1]
                        message += f' {value:.4f}, '
                    print(message[:-2])
                    print('='* 10)
                    with open(log_txt, 'a') as f:
                        f.write(message[:-2])
                        f.write('='* 10 + '\n')
                        tmp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        f.write(f'end time: {tmp_time}\n')
                    if args.wandb_log:
                        wandb.log(log_test)

                monitor_metric = f'val/' + monitor
                if log_val[monitor_metric] > best_val_metric:
                    is_better = True
                    best_val_metric = log_val[monitor_metric]

                if is_better:
                    test_metric.append(log_test.copy())

                if is_better:
                    best_test_metric = test_metric.copy()
                
                if master_process:
                    
                    if is_better:
                        checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'epoch': epoch,
                        }
                        for key, value in test_metric[-1].items():
                            checkpoint[key] = value
                            if 'pearsonr' in key:
                                rvalue = value
                        # print(f"saving checkpoint to {checkpoint_out_dir}")
                        # with open (log_txt, 'a') as f:
                        #     f.write(f"saving checkpoint to {checkpoint_out_dir}\n")
                        torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt_{args.session}_{rvalue:.4f}.pt'))
            if master_process:
                # test
                checkpoint = torch.load(os.path.join(checkpoint_out_dir, f'ckpt_{args.session}_{rvalue:.4f}.pt'), map_location=device)
                state_dict = checkpoint['model']
                if ddp:
                    # Assume that your model is wrapped with DDP, and add the `'module.'` prefix to each key.
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict['module.' + k] = v

                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)
                
                log = evaluate(model, data_loader_test, metrics)
                log_test = {k.replace('val', 'test'): v for k, v in log.items()}
                if master_process:
                    print('='* 10)
                    with open (log_txt, 'a') as f:
                        f.write('\n\n='* 10 + '\n')
                    message = f'###Final Test EEG: '
                    for key, value in log_test.items():
                        message += key.split('/')[-1]
                        message += f' {value:.4f}, '
                    print(message[:-2])
                    print('='* 10)
                    with open(log_txt, 'a') as f:
                        f.write(message[:-2])
                        f.write('='* 10 + '\n')
                        tmp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        f.write(f'end time: {tmp_time}\n')
                    if args.wandb_log:
                        wandb.log(log_test)

                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt_final_{args.session}_{rvalue:.4f}.pt'))
                
                    # if (epoch + 1) % args.save_ckpt_freq == 0:
                    #     print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                    #     with open(log_txt, 'a') as f:
                    #         f.write(f"saving checkpoint {epoch} to {checkpoint_out_dir}\n")
                    #     torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))
            
            if master_process:
                for test_metric in best_test_metric:
                    for key, value in test_metric.items():
                        m= key.split('/')[0][5:]
                        break
                    message = f'Best test metrics {m}:'
                    for key, value in test_metric.items():
                        message += key.split("/")[-1]
                        message += f' {value:.5f},'
                    print(message[:-2])
                    with open(log_txt, 'a') as f:
                        f.write(message[:-2] + '\n')

    if ddp:
        destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--out_dir', default='./result/', help='path where to save, empty for no saving')
    # parser.add_argument('--dataset_dir', default=r'X:\xxxxx\Data\MoBI\MoBI_npy_ss\\', help='path where data is')
    parser.add_argument('--dataset_dir', default=r'X:\xxxxx\Data\GPP\trialwise_dataset\normal\\', help='path where data is')
    # parser.add_argument('--pretrained_dir', default='checkpoints/MSM', help='path where pretrained model is')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--dataset', default='GPP', help='dataset: GPP')
    parser.add_argument('--loss_name', default='mse', help='loss name: mse, freq_mse, reward_mse, freq_reward_mse')


    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='GPP')
    parser.add_argument('--wandb_runname', default='GPP')
    parser.add_argument('--wandb_api_key', type=str)
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', default=2, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--block_size', default=512, type=int)

    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=False, action='store_true')
    parser.add_argument('--start_name', default='1', help='subject to start')
    parser.add_argument('--model_name', default='GaitGraph', help='model to use')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model_name = args.model_name
    
    start_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    ch_names = [
        'Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
        'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2',
        'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
        'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
        'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4',
        'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2'
    ]

    peri_remove = 0
    channel_neighbors = {
            # order: up-down-left-right
            "Fp1": ["AF7", "AF3", "AFz"],
            "Fp2": ["AF4", "AF8", "AFz"],
            "AF7": ["Fp1", "AF3", "F7"],
            "AF3": ["Fp1", "F3", "AF7", "AFz"],
            "AFz": ["Fp1", 'Fp2', "Fz", "AF3", "AF4"],
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
            "CP6": ["C6", "P6", "CP6", "TP8"],
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
    adj_matrix = np.zeros((num_channels, num_channels))

    for channel, neighbors in channel_neighbors.items():
        for neighbor in neighbors:
            i = ch_names.index(channel)
            j = ch_names.index(neighbor)

            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    np.fill_diagonal(adj_matrix, 1)
    args.seed = 256

    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float).to(device)

    config = load_config('./config.yaml')
    config.eegnet.adj_matrix = adj_matrix_tensor
    config.eegnet.eeg.time_step = 200
    args.batch_size = args.batch_size // torch.cuda.device_count()
    config.eegnet.loss_name = args.loss_name
    os.makedirs('./results', exist_ok=True)
    log_txt = f'./results/{model_name}_{args.dataset}_{start_time}_batch{args.batch_size}.txt'
    if model_name != "GaitGraph":
        args.epochs = 30
        args.batch_size = 64 // torch.cuda.device_count()
    with open(log_txt, 'a') as f:
        f.write(f"Available GPUs on this node: {torch.cuda.device_count()}\n")
        f.write(f"batch size: {args.batch_size}\n")
        f.write(f"loss name: {args.loss_name}\n")
    
    main(args)
