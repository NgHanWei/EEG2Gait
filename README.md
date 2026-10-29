A unified training, evaluation, and visualization pipeline for EEG-based gait decoding using contrastive learning, multi-head architectures, and strong baselines.
Supports FBM, GPP, and MoBI datasets with single-GPU, multi-GPU, and multi-node DDP training.

## Citation

If you use this work in your research, please cite the original paper:

Xi Fu, Rui Liu, Aung Aung Phyo Wai, Hannah Pulferer, Neethu Robinson, Gernot Müller-Putz, Cuntai Guan, *EEG2GAIT: A Hierarchical Graph Convolutional Network for EEG-Based Gait Decoding,*IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 34, pp. 441-454, 2026, doi: 10.1109/TNSRE.2025.3647101.

```bibtex
@article{fu2025eeg2gait,
  author  = {Fu, Xi and Liu, Rui and Wai, Aung Aung Phyo and
             Pulferer, Hannah and Robinson, Neethu and
             Müller-Putz, Gernot R. and Guan, Cuntai},
  title   = {EEG2GAIT: A Hierarchical Graph Convolutional
             Network for EEG-Based Gait Decoding},
  journal = {IEEE Transactions on Neural Systems and
             Rehabilitation Engineering},
  volume  = {34},
  pages   = {441--454},
  year    = {2025},
  doi     = {10.1109/TNSRE.2025.3647101,
  url     = {https://doi.org/10.1109/TNSRE.2025.3647101}
}
```

📂 Repository Structure
. \
├── train_baseline.py \
├── sbj_dep_train.py \
├── vis_embd.py \
├── cal_metrics.py \
├── config.yaml \
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
