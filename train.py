import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

import config
from src.model import PeptideESMOnlyModel, PepScopeModel
from src.dataset import PeptideDataset, MultiModalDataset, collate_fn, collate_mm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
LR = config.LR
WEIGHT_DECAY = config.WEIGHT_DECAY
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = config.BEST_MODEL_PATH
NUM_CLASSES = config.NUM_CLASSES
USE_MM = config.USE_MULTI_MODAL


class FocalLoss(nn.Module):
    def __init__(self, gamma=config.FOCAL_LOSS_GAMMA, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


def compute_class_alpha(labels, strategy="median"):
    num_classes = labels.shape[1]
    pos_count = labels.sum(axis=0)
    total = len(labels)
    if strategy == "median":
        median = np.median(pos_count)
        alpha = np.where(pos_count > 0, median / pos_count, 1.0)
        alpha = np.clip(alpha, 0.1, 0.99)
    else:
        alpha = 1.0 - (pos_count / total)
    return torch.tensor(alpha, dtype=torch.float32).to(DEVICE)


def to_device(batch, device=DEVICE):
    return tuple(x.to(device) for x in batch)


def train_one_epoch_mm(model, loader, criterion, optimizer, scaler):
    """Multi-modal training step."""
    model.train()
    total_loss = 0.0
    all_pred, all_label = [], []
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        seq, fp, nf, adj, mask, esm, label = to_device(batch)
        optimizer.zero_grad()
        with autocast():
            out = model(seq=seq, fp=fp, node_feat=nf, adj=adj,
                        esm_emb=esm, mask=mask)
            loss = criterion(out, label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        all_pred.append(out.detach().cpu().numpy())
        all_label.append(label.cpu().numpy())
        pbar.set_postfix(loss="%.4f" % loss.item())
    return total_loss / len(loader), np.concatenate(all_pred), np.concatenate(all_label)


@torch.no_grad()
def val_one_epoch_mm(model, loader, criterion):
    """Multi-modal validation step."""
    model.eval()
    total_loss = 0.0
    all_pred, all_label = [], []
    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        seq, fp, nf, adj, mask, esm, label = to_device(batch)
        out = model(seq=seq, fp=fp, node_feat=nf, adj=adj,
                    esm_emb=esm, mask=mask)
        loss = criterion(out, label)
        total_loss += loss.item()
        all_pred.append(out.cpu().numpy())
        all_label.append(label.cpu().numpy())
        pbar.set_postfix(loss="%.4f" % loss.item())
    return total_loss / len(loader), np.concatenate(all_pred), np.concatenate(all_label)


def train_one_epoch_esm(model, loader, criterion, optimizer, scaler):
    """ESM-only training step (legacy)."""
    model.train()
    total_loss = 0.0
    all_pred, all_label = [], []
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        esm, gx, ge, ga, gb, label = to_device(batch)
        optimizer.zero_grad()
        with autocast():
            out = model(esm, gx, ge, ga, gb)
            loss = criterion(out, label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        all_pred.append(out.detach().cpu().numpy())
        all_label.append(label.cpu().numpy())
        pbar.set_postfix(loss="%.4f" % loss.item())
    return total_loss / len(loader), np.concatenate(all_pred), np.concatenate(all_label)


@torch.no_grad()
def val_one_epoch_esm(model, loader, criterion):
    """ESM-only validation step (legacy)."""
    model.eval()
    total_loss = 0.0
    all_pred, all_label = [], []
    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        esm, gx, ge, ga, gb, label = to_device(batch)
        out = model(esm, gx, ge, ga, gb)
        loss = criterion(out, label)
        total_loss += loss.item()
        all_pred.append(out.cpu().numpy())
        all_label.append(label.cpu().numpy())
        pbar.set_postfix(loss="%.4f" % loss.item())
    return total_loss / len(loader), np.concatenate(all_pred), np.concatenate(all_label)


def run_train(train_data, val_data):
    if USE_MM:
        train_esm, train_seq, train_fp, train_graph, train_label, train_weights = train_data
        val_esm, val_seq, val_fp, val_graph, val_label = val_data
    else:
        train_esm, train_graph, train_label, train_weights = train_data
        val_esm, val_graph, val_label = val_data

    num_classes = train_label.shape[1]

    # Build dataset
    if USE_MM:
        train_ds = MultiModalDataset(train_seq, train_fp, train_graph,
                                     train_esm, train_label)
        val_ds = MultiModalDataset(val_seq, val_fp, val_graph,
                                   val_esm, val_label)
        collate = collate_mm
    else:
        train_ds = PeptideDataset(train_esm, train_graph, train_label)
        val_ds = PeptideDataset(val_esm, val_graph, val_label)
        collate = collate_fn

    # Balanced sampler
    if config.BALANCED_SAMPLER:
        sampler = WeightedRandomSampler(
            weights=train_weights, num_samples=len(train_weights),
            replacement=True)
        train_loader = DataLoader(
            train_ds, BATCH_SIZE, sampler=sampler,
            collate_fn=collate, num_workers=config.NUM_WORKERS)
        print("Using WeightedRandomSampler (balanced batches)")
    else:
        train_loader = DataLoader(
            train_ds, BATCH_SIZE, shuffle=True,
            collate_fn=collate, num_workers=config.NUM_WORKERS)

    val_loader = DataLoader(
        val_ds, BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=config.NUM_WORKERS)

    # Create model
    if USE_MM:
        model = PepScopeModel(
            num_classes=num_classes, esm_dim=train_esm.shape[1],
        ).to(DEVICE)
        print("Using PepScopeModel (multi-modal: seq + fp + GCN + ESM)")
        print("  Params: %d" % sum(p.numel() for p in model.parameters()))
    else:
        model = PeptideESMOnlyModel(
            esm_dim=train_esm.shape[1], num_classes=num_classes,
        ).to(DEVICE)
        print("Using PeptideESMOnlyModel (ESM-only)")

    # Loss
    alpha = compute_class_alpha(train_label, strategy=config.CLASS_WEIGHT_TYPE)
    print("Focal Loss alpha:", alpha.cpu().numpy().round(3))
    criterion = FocalLoss(gamma=config.FOCAL_LOSS_GAMMA, alpha=alpha)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    # Select train/val functions
    train_fn = train_one_epoch_mm if USE_MM else train_one_epoch_esm
    val_fn = val_one_epoch_mm if USE_MM else val_one_epoch_esm

    best_val_f1 = 0.0
    patience, wait = config.PATIENCE, 0

    for epoch in range(1, EPOCHS + 1):
        print()
        print("==== Epoch %d/%d ====" % (epoch, config.EPOCHS))
        train_loss, _, _ = train_fn(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_logits, val_label_np = val_fn(model, val_loader, criterion)
        scheduler.step()

        val_probs = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
        best_thresholds = []
        for i in range(num_classes):
            best_f1, best_t = 0, 0.5
            for t in range(5, 60):
                thr = t / 100
                y_bin = (val_probs[:, i] >= thr).astype(float)
                f1 = f1_score(val_label_np[:, i], y_bin, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, thr
            best_thresholds.append(best_t)

        y_pred = np.zeros_like(val_probs)
        for i in range(num_classes):
            y_pred[:, i] = (val_probs[:, i] >= best_thresholds[i]).astype(float)
        macro_f1 = f1_score(val_label_np, y_pred, average="macro", zero_division=0)

        per_f1 = []
        for i in range(num_classes):
            f1_i = f1_score(val_label_np[:, i], y_pred[:, i], zero_division=0)
            per_f1.append(f1_i)
        print("Loss: %.4f/%.4f | Macro F1: %.4f" % (train_loss, val_loss, macro_f1))
        print("  Per-class: %s" % " ".join(
            ["%s=%.3f" % (config.CATEGORY_NAMES[i][:6], per_f1[i])
             for i in range(num_classes)]))

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            wait = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_thresholds": best_thresholds,
            }, BEST_MODEL_PATH)
            print("  >> Best saved (F1=%.4f)" % best_val_f1)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping at epoch %d" % epoch)
                break

    print()
    print("Done! Best val Macro F1: %.4f" % best_val_f1)

