"""
Inference pipeline for the trained peptide classification model.
Loads model + ESM-2 once, then provides predict() for single/batch sequences.
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import PepScopeModel
from src.utils import encode_sequence, get_fingerprint, build_mol_graph

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


class PeptidePredictor:
    """Singleton predictor that loads model + ESM once and caches them."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.BEST_MODEL_PATH
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.esm_model = None
        self.thresholds = None
        self.cat_names = config.CATEGORY_NAMES
        self.num_classes = config.NUM_CLASSES
        self._load()

    def _load(self):
        print("[Inference] Loading model from %s" % self.model_path)
        checkpoint = torch.load(self.model_path, map_location=DEVICE)
        self.thresholds = checkpoint.get("best_thresholds", [0.5] * self.num_classes)

        self.model = PepScopeModel(num_classes=self.num_classes).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print("[Inference] Model loaded (%.2fK params)" % (
            sum(p.numel() for p in self.model.parameters()) / 1000))

        # Load ESM-2 (lazy, on first predict)
        self.tokenizer = None
        self.esm_model = None

    def _load_esm(self):
        if self.esm_model is not None:
            return
        print("[Inference] Loading ESM-2 model...")
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(config.ESM_MODEL)
        self.esm_model = AutoModel.from_pretrained(config.ESM_MODEL).to(DEVICE)
        self.esm_model.eval()
        print("[Inference] ESM-2 ready")

    def _extract_esm(self, seqs):
        """Batch ESM embedding extraction."""
        self._load_esm()
        all_e = []
        bs = 32
        for i in range(0, len(seqs), bs):
            batch = seqs[i:i+bs]
            inp = self.tokenizer(batch, padding=True, return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.no_grad():
                o = self.esm_model(**inp)
                mask = inp["attention_mask"].unsqueeze(-1).float()
                emb = (o.last_hidden_state * mask).sum(1) / mask.sum(1)
                all_e.append(emb.cpu().numpy())
        return np.concatenate(all_e).astype(np.float32)

    def validate_seq(self, seq):
        """Check if a sequence is valid. Returns (is_valid, error_msg)."""
        seq = seq.strip().upper()
        if not seq:
            return False, "Empty sequence"
        if len(seq) < 5:
            return False, "Sequence too short (<5 AA)"
        if len(seq) > 200:
            return False, "Sequence too long (>200 AA)"
        bad_aas = [c for c in seq if c not in VALID_AA]
        if bad_aas:
            return False, "Invalid amino acids: %s" % "".join(set(bad_aas))
        return True, ""

    @torch.no_grad()
    def predict(self, seqs):
        """
        Predict one or more sequences.
        Args:
            seqs: str or list of str
        Returns:
            list of dict: [{"seq": ..., "probs": {...}, "predicted": [...], "max_class": ...}, ...]
        """
        single = isinstance(seqs, str)
        if single:
            seqs = [seqs]

        seqs = [s.strip().upper() for s in seqs]

        # Filter valid
        valid_mask = []
        valid_seqs = []
        for s in seqs:
            ok, _ = self.validate_seq(s)
            valid_mask.append(ok)
            if ok:
                valid_seqs.append(s)

        if not valid_seqs:
            results = []
            for s in seqs:
                results.append({"seq": s, "error": "Invalid sequence", "probs": None})
            return results[0] if single else results

        # Preprocess: seq encoding + fingerprint + graph
        seq_enc = np.array([encode_sequence(s, config.SEQ_MAX_LEN)
                            for s in valid_seqs], dtype=np.int64)
        fp_enc = np.array([get_fingerprint(s, config.FP_DIM).astype(np.int64)
                           for s in valid_seqs], dtype=np.int64)
        graph_data = [build_mol_graph(s, max_atoms=config.MAX_ATOMS)
                      for s in valid_seqs]

        # ESM embedding
        esm_emb = self._extract_esm(valid_seqs)

        # Build tensors
        seq_t = torch.tensor(seq_enc, dtype=torch.long).to(DEVICE)
        fp_t = torch.tensor(fp_enc, dtype=torch.long).to(DEVICE)
        esm_t = torch.tensor(esm_emb, dtype=torch.float32).to(DEVICE)

        # Build graph batch
        B = len(valid_seqs)
        N = config.MAX_ATOMS
        nf = np.zeros((B, N, config.NODE_DIM), dtype=np.float32)
        adj = np.zeros((B, N, N), dtype=np.float32)
        mask = np.zeros((B, N), dtype=bool)
        for i, (node_feat, edge_pairs) in enumerate(graph_data):
            nf[i] = node_feat.astype(np.float32)
            mask[i] = (node_feat.sum(axis=-1) > 0)
            if len(edge_pairs) > 0:
                adj[i, edge_pairs[:, 0], edge_pairs[:, 1]] = 1.0
                adj[i, edge_pairs[:, 1], edge_pairs[:, 0]] = 1.0

        nf_t = torch.tensor(nf, dtype=torch.float32).to(DEVICE)
        adj_t = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
        mask_t = torch.tensor(mask, dtype=torch.bool).to(DEVICE)

        # Run model
        logits = self.model(seq=seq_t, fp=fp_t, node_feat=nf_t,
                            adj=adj_t, esm_emb=esm_t, mask=mask_t)
        probs = torch.sigmoid(logits).cpu().numpy()

        # Build results
        results = []
        vi = 0
        for i, ok in enumerate(valid_mask):
            if not ok:
                results.append({"seq": seqs[i], "error": "Invalid sequence",
                                "probs": None})
            else:
                p = probs[vi]
                pred = (p >= np.array(self.thresholds)).astype(int).tolist()
                max_idx = int(p.argmax())
                max_prob = float(p[max_idx])
                probs_dict = {}
                for j, name in enumerate(self.cat_names):
                    probs_dict[name] = float(round(p[j], 4))
                results.append({
                    "seq": seqs[i],
                    "probs": probs_dict,
                    "predicted": pred,
                    "max_class": self.cat_names[max_idx],
                    "max_prob": max_prob,
                    "thresholds": {self.cat_names[j]: float(self.thresholds[j])
                                   for j in range(self.num_classes)},
                })
                vi += 1

        return results[0] if single else results


# Global singleton
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = PeptidePredictor()
    return _predictor


def predict(seqs):
    return get_predictor().predict(seqs)

