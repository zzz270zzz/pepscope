import torch
import numpy as np
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """
    Dataset for MVMR-BPC style multi-modal model.
    Graph stored as (node_feat_padded, edge_pairs).
    node_feat loaded as float16, converted to float32 for model.
    """
    def __init__(self, seq_enc, fp_enc, graph_list, esm_emb, label_data):
        self.seq = torch.tensor(seq_enc, dtype=torch.long)
        self.fp = torch.tensor(fp_enc, dtype=torch.long)
        self.graph_list = graph_list  # list of (node_feat, edge_pairs)
        self.esm = torch.tensor(esm_emb, dtype=torch.float32)
        self.label = torch.tensor(label_data, dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        node_feat, edge_pairs = self.graph_list[idx]
        mask = (node_feat.sum(axis=-1) > 0)
        return (self.seq[idx],
                self.fp[idx],
                torch.tensor(node_feat, dtype=torch.float32),  # float16->float32
                torch.tensor(edge_pairs, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool),
                self.esm[idx],
                self.label[idx])


class PeptideDataset(Dataset):
    """Legacy ESM+Graph dataset."""
    def __init__(self, esm_embeddings, graph_list, label_data):
        self.esm = torch.tensor(esm_embeddings, dtype=torch.float32)
        self.graph_list = graph_list
        self.label = torch.tensor(label_data, dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        item = self.graph_list[idx]
        if isinstance(item, tuple) and len(item) >= 3:
            x, edge_index, edge_attr = item[:3]
        else:
            x = item
            edge_index = np.array([[0], [0]], dtype=np.int64)
            edge_attr = np.zeros((1, 4), dtype=np.float32)
        return (self.esm[idx],
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(edge_index, dtype=torch.long),
                torch.tensor(edge_attr, dtype=torch.float32),
                self.label[idx])


def collate_fn(batch):
    """Collate for PeptideDataset (legacy)."""
    esm_list, x_list, edge_list, edge_attr_list, label_list = zip(*batch)
    esm = torch.stack(esm_list)
    label = torch.stack(label_list)
    total_nodes = 0
    all_x, all_edge, all_edge_attr, batch_idx = [], [], [], []
    for i, (x, edge, attr) in enumerate(zip(x_list, edge_list, edge_attr_list)):
        n = x.shape[0]
        all_x.append(x)
        if edge.numel() > 0:
            all_edge.append(edge + total_nodes)
        else:
            all_edge.append(torch.zeros((2, 1), dtype=torch.long))
        all_edge_attr.append(attr)
        batch_idx.extend([i] * n)
        total_nodes += n
    return (esm,
            torch.cat(all_x, dim=0),
            torch.cat(all_edge, dim=1),
            torch.cat(all_edge_attr, dim=0),
            torch.tensor(batch_idx, dtype=torch.long),
            label)


def collate_mm(batch):
    """
    Collate for MultiModalDataset.
    Builds adjacency matrix from edge_pairs on-the-fly.
    """
    seq, fp, nf, ep, mask, esm, label = zip(*batch)

    seq_s = torch.stack(seq)
    fp_s = torch.stack(fp)
    nf_s = torch.stack(nf)  # float16 already converted to float32 in __getitem__
    mask_s = torch.stack(mask)
    esm_s = torch.stack(esm)
    label_s = torch.stack(label)

    B = len(ep)
    N = nf_s.shape[1]  # max_atoms
    adj = torch.zeros(B, N, N, dtype=torch.float32)
    for i in range(B):
        pairs = ep[i]
        if pairs.numel() > 0:
            adj[i, pairs[:, 0], pairs[:, 1]] = 1.0
            adj[i, pairs[:, 1], pairs[:, 0]] = 1.0

    return seq_s, fp_s, nf_s, adj, mask_s, esm_s, label_s
