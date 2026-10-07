import pkgutil
import deepchem as dc
import pandas as pd

LOADING_FUNCTIONS = {
    'bace_c': dc.molnet.load_bace_classification,
    'bace_r': dc.molnet.load_bace_regression,
    'bbbp': dc.molnet.load_bbbp,
    'clintox': dc.molnet.load_clintox,
    'hiv': dc.molnet.load_hiv,
    'sider': dc.molnet.load_sider,
    'tox21': dc.molnet.load_tox21,
    'esol': dc.molnet.load_delaney,
    'freesolv': dc.molnet.load_freesolv,
    'lipo': dc.molnet.load_lipo,
}

def load_datsets(name):

    if name not in LOADING_FUNCTIONS:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(LOADING_FUNCTIONS.keys())}")

    tasks , datasets , transformers = LOADING_FUNCTIONS[name](
        featurizer="raw",
        split = "scaffold",
    )

    train , valid , test = datasets

    return train , valid , test

