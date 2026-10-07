import pkgutil
import deepchem as dc
import pandas as pd

LOADING_FUNCTIONS = {
    'bace_c': dc.molnet.load_bace_classification,
    'bace_r': dc.molnet.load_bace_regression,
    'bbbp': dc.molnet.load_bbbp,
    'chembl': dc.molnet.load_chembl,
    'clearance': dc.molnet.load_clearance,
    'clintox': dc.molnet.load_clintox,
    'delaney': dc.molnet.load_delaney,
    'factors': dc.molnet.load_factors,
    'hiv': dc.molnet.load_hiv,
    'hopv': dc.molnet.load_hopv,
    'hppb': dc.molnet.load_hppb,
    'kaggle': dc.molnet.load_kaggle,
    'kinase': dc.molnet.load_kinase,
    'lipo': dc.molnet.load_lipo,
    'muv': dc.molnet.load_muv,
    'nci': dc.molnet.load_nci,
    'pcba': dc.molnet.load_pcba,
    'pdbbind': dc.molnet.load_pdbbind,
    'ppb': dc.molnet.load_ppb,
    'qm7': dc.molnet.load_qm7,
    'qm8': dc.molnet.load_qm8,
    'qm9': dc.molnet.load_qm9,
    'sampl': dc.molnet.load_sampl,
    'sider': dc.molnet.load_sider,
    'thermosol': dc.molnet.load_thermosol,
    'tox21': dc.molnet.load_tox21,
    'toxcast': dc.molnet.load_toxcast,
    'uv': dc.molnet.load_uv,
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

