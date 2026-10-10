import deepchem as dc
from rdkit import Chem
import types
import numpy as np
import pandas as pd

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
    'clearance':dc.molnet.load_clearance
}


def _unified_df(self):
    """Same columns (X, y, w, ids) for single- and multi-task datasets.
    Multi-task: y and w cells are arrays of length n_tasks.
    Single-task: y and w cells are plain scalars (same as DeepChem's default)."""
    n = len(self)
    y = np.asarray(self.y).reshape(n, -1)
    w = np.asarray(self.w).reshape(n, -1)
    multi = y.shape[1] > 1
    return pd.DataFrame({
        'X':   list(self.X),
        'y':   list(y) if multi else y[:, 0],
        'w':   list(w) if multi else w[:, 0],
        'ids': list(self.ids),
    })

def load_datsets(name, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    if name not in LOADING_FUNCTIONS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADING_FUNCTIONS)}")

    tasks, (dataset,), _ = LOADING_FUNCTIONS[name](
        featurizer=dc.feat.DummyFeaturizer(),
        splitter=None,
        transformers=[],
        reload=False,
    )

    keep = [i for i, smi in enumerate(dataset.ids)
            if Chem.MolFromSmiles(str(smi)) is not None]
    if len(keep) < len(dataset):
        print(f"[{name}] dropped {len(dataset) - len(keep)} unparsable molecules")
    dataset = dataset.select(keep)

    train, valid, test = dc.splits.ScaffoldSplitter().train_valid_test_split(
        dataset, frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test)

    # Make .to_dataframe() return the unified layout
    for ds in (train, valid, test):
        ds.to_dataframe = types.MethodType(_unified_df, ds)

    return train, valid, test
