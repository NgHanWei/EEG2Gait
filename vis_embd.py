import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from types import SimpleNamespace

from models.contraDCN import EEG_pred_multihead
from utils.utils import prepare_GPP_dataset_vis

# ----------------------------------------------------------------------
#                           Matplotlib 配置
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif"
})
plt.rcParams["font.size"] = 18


# ----------------------------------------------------------------------
#                          t-SNE 可视化函数
# ----------------------------------------------------------------------
def visualize_tsne_with_pca(embeds, phase, save_path="tsne_pca.png"):
    """单模型 embedding 的 t-SNE 可视化（PCA->t-SNE）"""

    # Convert to numpy
    if isinstance(embeds, torch.Tensor):
        embeds = embeds.detach().cpu().numpy()
    if isinstance(phase, torch.Tensor):
        phase = phase.detach().cpu().numpy()

    # Remove invalid phase
    mask = phase != -1
    embeds = embeds[mask]
    phase = phase[mask]

    # PCA
    pca_dim = min(10, embeds.shape[0])
    pca = PCA(n_components=pca_dim, random_state=42)
    embeds_pca = pca.fit_transform(embeds)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, embeds.shape[0] // 2)
    )
    embeds_tsne = tsne.fit_transform(embeds_pca)

    # Colors
    phase_colors = {0: "red", 1: "blue", 2: "orange", 3: "green"}
    phase_names = {0: "1", 1: "2", 2: "3", 3: "4"}

    plt.figure(figsize=(8, 6))
    for p in phase_colors:
        mask = (phase == p)
        plt.scatter(
            embeds_tsne[mask, 0], embeds_tsne[mask, 1],
            c=phase_colors[p], s=10, alpha=0.7, label=phase_names[p]
        )

    plt.title("t-SNE (after PCA) of EEG Embeddings")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


def visualize_tsne_compare(before_embed, before_phase, after_embed, after_phase,
                           save_path="tsne_compare.png"):
    """上下对比图：Before Stage I vs After Stage I"""

    def prepare_tsne(embeds, phase):
        if isinstance(embeds, torch.Tensor):
            embeds = embeds.detach().cpu().numpy()
        if isinstance(phase, torch.Tensor):
            phase = phase.detach().cpu().numpy()

        mask = phase != -1
        embeds = embeds[mask]
        phase = phase[mask]

        pca_dim = min(10, embeds.shape[0])
        pca = PCA(n_components=pca_dim, random_state=42)
        embeds_pca = pca.fit_transform(embeds)

        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, embeds.shape[0] // 2)
        )
        embeds_tsne = tsne.fit_transform(embeds_pca)
        return embeds_tsne, phase

    phase_colors = {0: "red", 1: "blue", 2: "orange", 3: "green"}
    phase_names = {0: "1", 1: "2", 2: "3", 3: "4"}

    tsne_b, phase_b = prepare_tsne(before_embed, before_phase)
    tsne_a, phase_a = prepare_tsne(after_embed, after_phase)

    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True)

    # BEFORE
    for p in phase_colors:
        mask = (phase_b == p)
        axs[0].scatter(tsne_b[mask, 0], tsne_b[mask, 1],
                       c=phase_colors[p], s=10, alpha=0.7, label=phase_names[p])
    axs[0].set_title("Before Stage I")
    axs[0].legend()
    axs[0].grid(True)

    # AFTER
    for p in phase_colors:
        mask = (phase_a == p)
        axs[1].scatter(tsne_a[mask, 0], tsne_a[mask, 1],
                       c=phase_colors[p], s=10, alpha=0.7, label=phase_names[p])
    axs[1].set_title("After Stage I")
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle("t-SNE Plot of EEG Embeddings from NeuroDyGait", fontsize=22, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


# ----------------------------------------------------------------------
#                   Extract Embeddings from Model
# ----------------------------------------------------------------------
def extract_embeddings(model, dataloader, device):
    """Extract embeddings with model(..., vis_embd=True)."""
    model.eval()

    all_embed, all_phase = [], []

    with torch.no_grad():
        for batch in dataloader:

            eeg = batch["EEG_X"]
            if isinstance(eeg, np.ndarray):
                eeg = torch.from_numpy(eeg)
            eeg = eeg.unsqueeze(1).float().to(device)

            phase = batch["phase"].to(device)

            # Mask phase== -1
            mask = (phase != -1)
            eeg = eeg[mask]
            phase = phase[mask]

            data = {
                "EEG": eeg,
                "phase": phase,
                "domain": batch.get("domain", None),
                "Y": batch.get("Y", None),
                "transitions": batch.get("transitions", None)
            }

            emb = model(data, vis_embd=True)
            all_embed.append(emb.cpu().numpy())
            all_phase.append(phase.cpu().numpy())

    return np.concatenate(all_embed, axis=0), np.concatenate(all_phase, axis=0)


# ----------------------------------------------------------------------
#                     Main — Model Loading + Visualization
# ----------------------------------------------------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_path = "./results/GPP_ind/ContraDCN_pred/ckpt_fold9_0.6727.pt"
    ckpt = torch.load(pretrained_path, map_location=device)
    args = SimpleNamespace(**ckpt["args"])
    state_dict = ckpt["model"]

    # ------------------ Dataset split (fixed) ------------------
    mode = "normal"

    if mode == "normal":
        val_idx = list(range(40, 45))
        test_idx = list(range(45, 50))
        sbjs = list(range(50))
    else:
        val_idx = test_idx = sbjs = [1]

    sessions = ["2", "1"]
    args.dataset_dir = "/home/fuxi/FT_dataset_PhaseTrans/"
    args.num_eeg_channels = 59
    args.n_domains = 100

    # session list
    train_sessions = [f"{j}_{s}" for j in sbjs for s in sessions]
    val_sessions   = [f"{i}_{s}" for i in val_idx for s in sessions]
    test_sessions  = [f"{i}_{s}" for i in test_idx for s in sessions]

    val_domains  = [int(v.split("_")[0]) * 2 + int(v.split("_")[1]) - 3 for v in val_sessions]
    test_domains = [int(t.split("_")[0]) * 2 + int(t.split("_")[1]) - 3 for t in test_sessions]
    target_domains = val_domains + test_domains

    # ------------------ Load model (AFTER stage I) ------------------
    model_after = EEG_pred_multihead(
        num_eeg_channels=args.num_eeg_channels,
        time_steps=args.time_steps,
        embed_dim=args.embed_dim,
        kernel_width=args.kernel_width,
        pool_width=args.pool_width,
        nFiltLaterLayer=args.nFiltLaterLayer,
        target_domains=target_domains,
    ).float().to(device)
    model_after.load_state_dict(state_dict)
    model_after.eval()

    # ------------------ Load dataset ------------------
    train_set, val_set, test_set = prepare_GPP_dataset_vis(
        args.dataset_dir, train_sessions, val_sessions, test_sessions
    )

    loader_train = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False)

    # ------------------------ Extract embeddings ------------------------
    print("Extracting AFTER-Stage-I embeddings...")
    after_embed, after_phase = extract_embeddings(model_after, loader_train, device)

    # ------------------------ BEFORE-Stage-I model ------------------------
    print("Extracting BEFORE-Stage-I embeddings...")
    model_before = EEG_pred_multihead(
        num_eeg_channels=args.num_eeg_channels,
        time_steps=args.time_steps,
        embed_dim=args.embed_dim,
        kernel_width=args.kernel_width,
        pool_width=args.pool_width,
        nFiltLaterLayer=args.nFiltLaterLayer,
        target_domains=target_domains,
    ).float().to(device)

    before_embed, before_phase = extract_embeddings(model_before, loader_train, device)

    # ---------------------- Visualize ----------------------
    visualize_tsne_with_pca(after_embed, after_phase, "tsne_after.png")
    visualize_tsne_compare(before_embed, before_phase, after_embed, after_phase, "tsne_compare.png")

    print("Visualization Done.")

