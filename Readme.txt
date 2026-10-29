A unified training, evaluation, and visualization pipeline for EEG-based gait decoding using contrastive learning, multi-head architectures, and strong baselines.
Supports FBM, GPP, and MoBI datasets with single-GPU, multi-GPU, and multi-node DDP training.

## Citation

If you use this work in your research, please cite the original paper:

Fu, X., Liu, R., Wai, A. A. P., Pulferer, H., Robinson, N., Müller-Putz, G. R., & Guan, C. *EEG2GAIT: A deep learning framework for decoding human gait dynamics from EEG signals.* IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE).


📂 Repository Structure
.
├── train_baseline.py
├── sbj_dep_train.py
├── vis_embd.py
├── cal_metrics.py
├── config.yaml
├── dataset folder

🚀 Key Features
Multi-dataset support (FBM / GPP / MoBI)
Distributed training (DDP: single GPU / multi-GPU / multi-node)
Subject-dependent evaluation
Embedding visualization (t-SNE / PCA)


🚀 Supported Models
EEG2GAIT

And other models
ContraDCN / ContraWR
SPaRCNet
FFCL
ST-Transformer
TSception
DeepConvNet
TCN
EEGConformer
CNNTransformer

🚀 Data Preprocessing
Preprocess the raw data (filter and partition) for MoBI
utils/MoBI_npy_dataset_build.py
Need to revise the data_path_folder to your own in line 165

Preprocess the raw data (filter and partition) for GPP 
utils/ft_dataset_build_ss.py
Need to revise the data_path_folder to your own in line 567

The processed data are stored as follows:
GPP/trialwise_dataset/normal: normal walking without sand bag
GPP/trialwise_dataset/left: walking with 1 sand bag on left foot
GPP/trialwise_dataset/right: walking with 1 sand bag on right foot
GPP/trialwise_dataset/left_2sand: walking with 2 sand bag on left foot

🚀 Training Scripts
Train model and test with subject-dependent setting for EEG2GAIT
sbj_dep_train.py

If you have one GPU, use the command:
python sbj_dep_train.py \
  --model_name GaitGraph \
  --batch_size 32 \
  --dataset "MoBI" \
  --loss_name 'freq_reward_mse' \
  --dataset_dir data_path_folder

## Dataset Loading
change the path directory at sbj_dep_train.py at the line containing args.dataset_dir

## Session Choosing
Change the line here            args.session = str(i) + '_' + str(j)
i: subject
j: session

Distributed Training
Multi-GPU (single node)
torchrun --standalone --nproc_per_node=4 sbj_dep_train.py

Multi-node
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=IP \
  --master_port=12345 \
  sbj_dep_train.py


Baseline Model Training
python train_baseline.py --model_name TSception

Embedding Visualization
python vis_embd.py --checkpoint path/to/ckpt.pt

Metrics for Gait prediction
python cal_metrics.py
Metrics implemented in cal_metrics.py:
R²
RMSE
MAE
Pearson correlation
Accuracy (if classification)
F1 / weighted F1


🚀 Dependencies
torch >= 2.0
numpy
scikit-learn
matplotlib
pyyaml
braindecode
dotmap
pyhealth
wandb
tqdm
