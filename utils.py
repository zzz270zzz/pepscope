import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
                "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb",
                "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
                "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr",
                "Cr", "Pt", "Hg", "Pb", "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        + one_of_k_encoding_unk(
            atom.GetValence(Chem.ValenceType.IMPLICIT),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        + [atom.GetIsAromatic()]
    )


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}:")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# ── GCN graph builder (float16 node_feat to save disk space) ──


def build_mol_graph(peptide_seq, max_atoms=100, node_dim=78):
    """
    Build GCN-compatible graph, storing edges as compact pairs.
    node_feat stored as float16 to save disk space.
    Returns (node_feat, edge_pairs) or zeros on failure.
    """
    try:
        mol = Chem.MolFromFASTA(peptide_seq)
        if mol is None:
            raise ValueError("mol is None")
        num_atoms = min(mol.GetNumAtoms(), max_atoms)

        feat_list = [atom_features(a) for a in mol.GetAtoms()[:num_atoms]]
        node_feat = np.zeros((max_atoms, node_dim), dtype=np.float16)
        node_feat[:num_atoms] = np.array(feat_list, dtype=np.float16)

        pairs = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < max_atoms and j < max_atoms:
                pairs.append([i, j])

        edge_pairs = np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)
        return node_feat, edge_pairs
    except Exception:
        return (
            np.zeros((max_atoms, node_dim), dtype=np.float16),
            np.zeros((0, 2), dtype=np.int64),
        )


def edges_to_adj(edge_pairs, N):
    """Convert edge pairs [E, 2] to symmetric adjacency matrix [N, N]."""
    adj = np.zeros((N, N), dtype=np.float32)
    if len(edge_pairs) > 0:
        adj[edge_pairs[:, 0], edge_pairs[:, 1]] = 1.0
        adj[edge_pairs[:, 1], edge_pairs[:, 0]] = 1.0
    return adj


def normalize_adj(adj_matrix):
    """D^(-1/2) @ A @ D^(-1/2) for each graph in batch."""
    d = np.maximum(adj_matrix.sum(axis=-1), 1e-10)
    d_inv_sqrt = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
    return d_inv_sqrt[..., None] * adj_matrix * d_inv_sqrt[:, None, :]


# ── Legacy PyG graph builder ──

BOND_TYPE_TO_IDX = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def build_peptide_graph(peptide_seq, node_dim=78):
    """Legacy: PyG edge-list format (node_feat, edge_index, edge_attr)."""
    try:
        mol = Chem.MolFromFASTA(peptide_seq)
        if mol is None:
            raise ValueError("mol is None")
        num_atoms = mol.GetNumAtoms()
        node_feat = np.zeros((num_atoms, node_dim), dtype=np.float32)
        for i, atom in enumerate(mol.GetAtoms()):
            feat = atom_features(atom)
            node_feat[i, : len(feat)] = feat
        edges, edge_attr = [], []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            feat = np.zeros(4, dtype=np.float32)
            feat[BOND_TYPE_TO_IDX.get(bt, 0)] = 1.0
            edges.append([u, v])
            edges.append([v, u])
            edge_attr.append(feat)
            edge_attr.append(feat)
        if not edges:
            edge_index = np.array([[0], [0]], dtype=np.int64)
            edge_attr = np.zeros((1, 4), dtype=np.float32)
        else:
            edge_index = np.array(edges, dtype=np.int64).T
            edge_attr = np.array(edge_attr, dtype=np.float32)
        return node_feat, edge_index, edge_attr
    except Exception:
        return (
            np.zeros((1, node_dim), dtype=np.float32),
            np.array([[0], [0]], dtype=np.int64),
            np.zeros((1, 4), dtype=np.float32),
        )


# ── Sequence encoding ──


def encode_sequence(peptide_seq, max_len=100):
    aa_vocab = {
        "<PAD>": 0, "A": 1, "R": 2, "N": 3, "D": 4, "C": 5,
        "E": 6, "Q": 7, "G": 8, "H": 9, "I": 10,
        "L": 11, "K": 12, "M": 13, "F": 14, "P": 15,
        "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20,
    }
    code = [aa_vocab.get(aa, 0) for aa in peptide_seq.upper()]
    if len(code) > max_len:
        code = code[:max_len]
    else:
        code += [0] * (max_len - len(code))
    return np.array(code, dtype=np.int64)


# ── Fingerprint (always float32) ──


def get_fingerprint(peptide_seq, fp_dim=2048):
    AA_TO_SMILES = {
        "A": "CC(C(=O)O)N", "R": "NCCCC(C(=O)O)N=C(N)N",
        "N": "NC(=O)CC(C(=O)O)N", "D": "O=C(O)CC(C(=O)O)N",
        "C": "SC(C(=O)O)N", "E": "O=C(O)CCC(C(=O)O)N",
        "Q": "NC(=O)CCC(C(=O)O)N", "G": "C(C(=O)O)N",
        "H": "NCC1=CN=CN1C(C(=O)O)N", "I": "CCC(C)C(C(=O)O)N",
        "L": "CC(C)CC(C(=O)O)N", "K": "NCCCCC(C(=O)O)N",
        "M": "CSCCC(C(=O)O)N", "F": "c1ccccc1CC(C(=O)O)N",
        "P": "C1CCN(C1)C(C(=O)O)", "S": "OCC(C(=O)O)N",
        "T": "CC(O)C(C(=O)O)N", "W": "c1ccc2c(c1)c(CN)c[nH]2CC(=O)O",
        "Y": "Oc1ccc(CC(C(=O)O)N)cc1", "V": "CC(C)C(C(=O)O)N",
    }
    try:
        smi_parts = [AA_TO_SMILES[aa] for aa in peptide_seq.upper()]
        full_smi = "".join(smi_parts)
        mol = Chem.MolFromSmiles(full_smi)
        if mol is None:
            return np.zeros(fp_dim, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_dim)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(fp_dim, dtype=np.float32)


def get_daylight_fingerprint(peptide_seq, fp_dim=2048):
    AA_TO_SMILES = {
        "A": "CC(C(=O)O)N", "R": "NCCCC(C(=O)O)N=C(N)N",
        "N": "NC(=O)CC(C(=O)O)N", "D": "O=C(O)CC(C(=O)O)N",
        "C": "SC(C(=O)O)N", "E": "O=C(O)CCC(C(=O)O)N",
        "Q": "NC(=O)CCC(C(=O)O)N", "G": "C(C(=O)O)N",
        "H": "NCC1=CN=CN1C(C(=O)O)N", "I": "CCC(C)C(C(=O)O)N",
        "L": "CC(C)CC(C(=O)O)N", "K": "NCCCCC(C(=O)O)N",
        "M": "CSCCC(C(=O)O)N", "F": "c1ccccc1CC(C(=O)O)N",
        "P": "C1CCN(C1)C(C(=O)O)", "S": "OCC(C(=O)O)N",
        "T": "CC(O)C(C(=O)O)N", "W": "c1ccc2c(c1)c(CN)c[nH]2CC(=O)O",
        "Y": "Oc1ccc(CC(C(=O)O)N)cc1", "V": "CC(C)C(C(=O)O)N",
    }
    try:
        smi_parts = [AA_TO_SMILES[aa] for aa in peptide_seq.upper()]
        full_smi = "".join(smi_parts)
        mol = Chem.MolFromSmiles(full_smi)
        if mol is None:
            return np.zeros(fp_dim, dtype=np.float32)
        fp = Chem.RDKFingerprint(mol, fpSize=fp_dim)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(fp_dim, dtype=np.float32)


def get_fused_fingerprint(peptide_seq, fp_dim=2048):
    ecfp = get_fingerprint(peptide_seq, fp_dim)
    dl = get_daylight_fingerprint(peptide_seq, fp_dim)
    return np.concatenate([ecfp, dl]).astype(np.int64)


def get_fused_fingerprint_float(peptide_seq, fp_dim=2048):
    ecfp = get_fingerprint(peptide_seq, fp_dim)
    dl = get_daylight_fingerprint(peptide_seq, fp_dim)
    return np.concatenate([ecfp, dl]).astype(np.float32)
