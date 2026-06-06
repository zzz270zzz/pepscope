import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

import config
from src.model import PeptideESMOnlyModel, PepScopeModel
from src.dataset import (
    PeptideDataset, MultiModalDataset,
    collate_fn, collate_mm
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.INFER_BATCH_SIZE
MODEL_PATH = config.BEST_MODEL_PATH
NUM_CLASSES = config.NUM_CLASSES
CAT_NAMES = config.CATEGORY_NAMES
USE_MM = config.USE_MULTI_MODAL


def to_device(batch, device=DEVICE):
    return tuple(x.to(device) for x in batch)


def test_model(test_data):
    if USE_MM:
        test_esm, test_seq, test_fp, test_graph, test_label = test_data
        test_ds = MultiModalDataset(test_seq, test_fp, test_graph,
                                    test_esm, test_label)
        collate = collate_mm
    else:
        test_esm, test_graph, test_label = test_data
        test_ds = PeptideDataset(test_esm, test_graph, test_label)
        collate = collate_fn

    num_classes = test_label.shape[1]
    test_loader = DataLoader(
        test_ds, BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=config.NUM_WORKERS)

    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    best_thresholds = checkpoint.get("best_thresholds", [0.5] * num_classes)

    if USE_MM:
        model = PepScopeModel(
            num_classes=num_classes, esm_dim=test_esm.shape[1],
        ).to(DEVICE)
    else:
        model = PeptideESMOnlyModel(
            esm_dim=test_esm.shape[1], num_classes=num_classes,
        ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_pred, all_label = [], []
    with torch.no_grad():
        for batch in test_loader:
            if USE_MM:
                seq, fp, nf, adj, mask, esm, label = to_device(batch)
                logits = model(seq=seq, fp=fp, node_feat=nf, adj=adj,
                               esm_emb=esm, mask=mask)
            else:
                esm, gx, ge, ga, gb, label = to_device(batch)
                logits = model(esm, gx, ge, ga, gb)
            all_pred.append(logits.cpu().numpy())
            all_label.append(label.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_label = np.concatenate(all_label)
    all_probs = 1.0 / (1.0 + np.exp(-all_pred))

    y_pred = np.zeros_like(all_probs)
    for i in range(num_classes):
        y_pred[:, i] = (all_probs[:, i] >= best_thresholds[i]).astype(float)

    macro_f1 = f1_score(all_label, y_pred, average="macro", zero_division=0)
    print("==== Test (per-class thresholds) ====")
    for i in range(num_classes):
        f1 = f1_score(all_label[:, i], y_pred[:, i], zero_division=0)
        prec = precision_score(all_label[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(all_label[:, i], y_pred[:, i], zero_division=0)
        try:
            auc = roc_auc_score(all_label[:, i], all_probs[:, i])
        except Exception:
            auc = 0.0
        name = CAT_NAMES[i] if i < len(CAT_NAMES) else ("Class" + str(i))
        print("  {name:>12s}: F1={f1:.3f} P={prec:.3f} R={rec:.3f} AUC={auc:.3f} thr={thr:.2f}".format(
            name=name, f1=f1, prec=prec, rec=rec, auc=auc, thr=best_thresholds[i]))

    print()
    print("  Macro F1: {:.4f}".format(macro_f1))

    # MVMR-BPC style multi-label metrics
    try:
        from evaluation import evaluate
        aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(y_pred, all_label)
        print()
        print("==== Multi-label Evaluation (MVMR-BPC style) ====")
        print("  Aiming:        %.4f" % aiming)
        print("  Coverage:      %.4f" % coverage)
        print("  Accuracy:      %.4f" % accuracy)
        print("  AbsoluteTrue:  %.4f" % absolute_true)
        print("  AbsoluteFalse: %.4f" % absolute_false)
    except Exception:
        pass

    return {"Macro_F1": macro_f1}

