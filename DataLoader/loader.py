import deepchem as dc
from rdkit import Chem

LOADING_FUNCTIONS = {
    'bace_classification': dc.molnet.load_bace_classification,
    'bace_regression': dc.molnet.load_bace_regression,
    'bbbp': dc.molnet.load_bbbp,
    'clintox': dc.molnet.load_clintox,
    'hiv': dc.molnet.load_hiv,
    'sider': dc.molnet.load_sider,
    'tox21': dc.molnet.load_tox21,
    'delaney': dc.molnet.load_delaney,
    'freesolv': dc.molnet.load_freesolv,
    'lipo': dc.molnet.load_lipo,
}

def load_datsets(name, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    if name not in LOADING_FUNCTIONS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADING_FUNCTIONS)}")

    # DummyFeaturizer keeps raw SMILES strings -> no RDKit parsing at load time.
    # splitter=None -> one unsplit dataset; we scaffold-split after filtering.
    tasks, (dataset,), _ = LOADING_FUNCTIONS[name](
        featurizer=dc.feat.DummyFeaturizer(),
        splitter=None,
        transformers=[],
        reload=False,
    )

    # Drop molecules RDKit can't parse (e.g. the Al valence-6 SMILES in Tox21)
    keep = [i for i, smi in enumerate(dataset.ids)
            if Chem.MolFromSmiles(str(smi)) is not None]
    dropped = len(dataset) - len(keep)
    if dropped:
        print(f"[{name}] dropped {dropped} unparsable molecules")
    dataset = dataset.select(keep)

    train, valid, test = dc.splits.ScaffoldSplitter().train_valid_test_split(
        dataset,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
    )
    return train, valid, test
