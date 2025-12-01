import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Crippen

from . import utils


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


def morgan_fingerprint(mol: Chem.Mol, radius: int, n_bits: int) -> List[int]:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(map(int, fp.ToBitString()))


def maccs_keys(mol: Chem.Mol) -> List[int]:
    fp = MACCSkeys.GenMACCSKeys(mol)
    return list(map(int, fp.ToBitString()))


def physchem_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    return {
        "desc_mol_wt": Descriptors.MolWt(mol),
        "desc_tpsa": Descriptors.TPSA(mol),
        "desc_hbd": Descriptors.NumHDonors(mol),
        "desc_hba": Descriptors.NumHAcceptors(mol),
        "desc_logp": Crippen.MolLogP(mol),
        "desc_num_rings": Descriptors.RingCount(mol),
        "desc_num_atoms": mol.GetNumAtoms(),
    }


def featurize_row(
    row: pd.Series,
    radius: int,
    n_bits: int,
    include_maccs: bool,
) -> Optional[Dict[str, float]]:
    smiles = row.get("CanonicalSMILES")
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    mfp_bits = morgan_fingerprint(mol, radius=radius, n_bits=n_bits)
    maccs_bits = maccs_keys(mol) if include_maccs else []
    descs = physchem_descriptors(mol)

    features: Dict[str, float] = {
        "query": row.get("query"),
        "cid": row.get("CID"),
        "smiles": smiles,
        "inchi": row.get("InChI"),
    }
    features.update({f"mfp_{i}": bit for i, bit in enumerate(mfp_bits)})
    if include_maccs:
        features.update({f"maccs_{i}": bit for i, bit in enumerate(maccs_bits)})
    features.update(descs)
    return features


def featurize_dataframe(df: pd.DataFrame, settings: Dict[str, Any], logger) -> pd.DataFrame:
    fp_cfg = settings.get("fingerprints", {}).get("morgan", {})
    maccs_cfg = settings.get("fingerprints", {}).get("maccs", {})
    radius = int(fp_cfg.get("radius", 2))
    n_bits = int(fp_cfg.get("n_bits", 2048))
    include_maccs = bool(maccs_cfg.get("enabled", True))

    logger.info("Featurizing %d molecules (radius=%d, bits=%d, MACCS=%s)", len(df), radius, n_bits, include_maccs)

    feature_rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        feats = featurize_row(row, radius=radius, n_bits=n_bits, include_maccs=include_maccs)
        if feats:
            feature_rows.append(feats)
        else:
            logger.warning("Skipping invalid SMILES: %s", row.get("CanonicalSMILES"))

    if not feature_rows:
        raise RuntimeError("No valid molecules to featurize")

    feat_df = pd.DataFrame(feature_rows)
    return feat_df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    non_feature_cols = {"query", "cid", "smiles", "inchi", "target", "ridge_pred", "rf_pred"}
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    return df[feature_cols]


def main() -> None:
    args = utils.parse_args()
    settings = utils.load_settings(args.config)
    logger = utils.get_logger(__name__)

    raw_path = settings.get("paths", {}).get("raw_data", "data/raw/compounds.csv")
    processed_path = settings.get("paths", {}).get("processed_data", "data/processed/processed.csv")
    utils.ensure_directories([os.path.dirname(processed_path)])

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run data_download first.")

    raw_df = pd.read_csv(raw_path)
    feat_df = featurize_dataframe(raw_df, settings=settings, logger=logger)
    feat_df.to_csv(processed_path, index=False)
    logger.info("Saved features to %s", processed_path)


if __name__ == "__main__":
    main()
