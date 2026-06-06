import torch
import torch.nn as nn
import torch.nn.functional as F


# ©¤©¤ PeptideESMOnlyModel ©¤©¤

class PeptideESMOnlyModel(nn.Module):
    """ESM-only baseline."""
    def __init__(self, esm_dim=640, num_classes=8, dropout_rate=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(esm_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, esm_emb, **kwargs):
        return self.classifier(esm_emb)


# ©¤©¤ Simple GCN Layer ©¤©¤

class SimpleGCNLayer(nn.Module):
    """GCN layer: H_out = ReLU(A_hat @ H @ W). Accepts pre-normalized adjacency."""
    def __init__(self, in_dim, out_dim, use_bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj_norm, mask=None):
        support = self.linear(x)
        out = torch.matmul(adj_norm, support)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        return F.relu(out)


class SimpleGCNEncoder(nn.Module):
    """2-layer GCN encoder with global mean pooling -> graph embedding."""
    def __init__(self, in_dim=78, hidden_dim=128, out_dim=64):
        super().__init__()
        self.conv1 = SimpleGCNLayer(in_dim, hidden_dim)
        self.conv2 = SimpleGCNLayer(hidden_dim, out_dim)

    def forward(self, node_feat, adj_norm, mask=None):
        x = self.conv1(node_feat, adj_norm, mask)
        x = self.conv2(x, adj_norm, mask)
        if mask is not None:
            valid = mask.unsqueeze(-1).float()
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return x


# ©¤©¤ Sequence Branch ©¤©¤

class SequenceBranch(nn.Module):
    """Dilated Conv1D x3 + BiLSTM + FC."""
    def __init__(self, vocab_size=21, embed_dim=100, dilation_rates=(2,4,8),
                 conv_filters=64, pool_size=5, lstm_units=80, fc_dim=128,
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        for d in dilation_rates:
            seq = nn.Sequential(
                nn.Conv1d(embed_dim, conv_filters, kernel_size=2,
                          padding=d//2, dilation=d),
                nn.MaxPool1d(kernel_size=pool_size, stride=1,
                             padding=pool_size//2),
            )
            self.convs.append(seq)
        self.bilstm = nn.LSTM(conv_filters * len(dilation_rates), lstm_units,
                              bidirectional=True, batch_first=True)
        self.fc = nn.Linear(lstm_units * 2, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(fc_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        conv_outs = []
        for conv in self.convs:
            out = conv(x)
            conv_outs.append(out)
        x = torch.cat(conv_outs, dim=1)
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


# ©¤©¤ Fingerprint Branch (FRN+TLU) ©¤©¤

class FRNLayer(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.tau = nn.Parameter(torch.full((1,), -0.1))

    def forward(self, x):
        nu2 = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x / (torch.sqrt(nu2) + self.eps)
        x = self.gamma * x + self.beta
        return torch.maximum(x, self.tau)


class TLULayer(nn.Module):
    def __init__(self, init_tau=0.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32))

    def forward(self, x):
        return torch.maximum(x, self.tau)


class FingerprintBranch(nn.Module):
    """Embed fingerprint bits -> 3xConv1D + FRN+TLU -> GAP -> FC."""
    def __init__(self, fp_len=2048, embed_dim=128, conv_channels=(16, 32, 32),
                 out_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(3, embed_dim)
        self.fpn = nn.Linear(embed_dim, conv_channels[0])
        self.conv_layers = nn.ModuleList()
        in_c = conv_channels[0]
        for out_c in conv_channels:
            block = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=5, padding="same"),
                FRNLayer(out_c),
                TLULayer(),
            )
            self.conv_layers.append(block)
            in_c = out_c
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_fc = nn.Linear(conv_channels[-1], out_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fpn(x)
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
        x = self.gap(x).squeeze(-1)
        x = self.out_fc(x)
        return x


# ---- PepScopeModel ----

def normalize_adj_pytorch(adj, mask=None):
    """D^(-1/2) @ A @ D^(-1/2) for batched adjacency matrices."""
    if mask is not None:
        adj = adj * mask.unsqueeze(1).float() * mask.unsqueeze(2).float()
    d = adj.sum(dim=-1).clamp(min=1e-10)
    d_inv_sqrt = torch.where(d > 1e-10, 1.0 / d.sqrt(), torch.zeros_like(d))
    return d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)


class PepScopeModel(nn.Module):
    """
    Multi-modal model (MVMR-BPC style):
      seq branch + fingerprint branch + GCN graph branch + ESM branch
      -> Concat -> classifier
    """
    def __init__(self, num_classes=8, seq_max_len=100, fp_len=2048,
                 esm_dim=640, node_dim=78, max_atoms=100,
                 seq_fc_dim=128, fp_out_dim=64, gcn_hidden=128,
                 gcn_out_dim=64, esm_proj_dim=64, dropout=0.3):
        super().__init__()
        self.max_atoms = max_atoms

        self.seq_branch = SequenceBranch(
            vocab_size=21, embed_dim=100, fc_dim=seq_fc_dim, dropout=dropout)
        self.fp_branch = FingerprintBranch(fp_len=fp_len, out_dim=fp_out_dim)
        self.graph_encoder = SimpleGCNEncoder(
            in_dim=node_dim, hidden_dim=gcn_hidden, out_dim=gcn_out_dim)
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, esm_proj_dim),
        )

        fusion_dim = seq_fc_dim + fp_out_dim + gcn_out_dim + esm_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(self, seq, fp, node_feat, adj, esm_emb, mask=None):
        seq_feat = self.seq_branch(seq)
        fp_feat = self.fp_branch(fp)
        adj_norm = normalize_adj_pytorch(adj, mask)
        graph_feat = self.graph_encoder(node_feat, adj_norm, mask)
        esm_feat = self.esm_proj(esm_emb)

        fused = torch.cat([seq_feat, fp_feat, graph_feat, esm_feat], dim=1)
        logits = self.classifier(fused)
        return logits


