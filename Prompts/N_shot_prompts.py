# Classification Prompts

BBBP_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule has blood-brain barrier penetration capability.
You will be provided with example molecules and their binary labels: Yes (penetrative) or No (non-penetrative).

Please predict the label for the given molecule and provide a probability score between 0.0 and 1.0 indicating the likelihood of it being penetrative (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


BACE_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule can inhibit Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1).
You will be provided with example molecules and their binary labels: Yes (inhibitor) or No (non-inhibitor).

Please predict the label for the given molecule and provide a probability score between 0.0 and 1.0 indicating the likelihood of it being an inhibitor (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


Tox21_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule is toxic.
You will be provided with example molecules and their binary labels: Yes (toxic) or No (not toxic).

Please predict the label for the given molecule and provide a probability score between 0.0 and 1.0 indicating the likelihood of it being toxic (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


HIV_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SELFIES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule can inhibit HIV replication.
You will be provided with example molecules and their binary labels: Yes (can inhibit) or No (cannot inhibit).

Please predict the label for the given molecule and provide a probability score between 0.0 and 1.0 indicating the likelihood of it being able to inhibit HIV replication (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


ClinTox_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule is clinically-trial-toxic or FDA approved.
You will be provided with example molecules and their binary labels: Yes (toxic/failed) or No (not toxic/approved).

Please predict the label for the given molecule and provide a probability score between 0.0 and 1.0 indicating the likelihood of it being toxic (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


SIDER_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule has specific adverse drug reactions or side effects.
You will be provided with example molecules and their binary labels: Yes (has side effect) or No (does not have side effect).

Please predict the label for the given molecule and provide a probability score between 0.0 and 1.0 indicating the likelihood of it having the side effect (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


# Regression Prompts

ESOL_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict the water solubility (log solubility in mols per litre) of the molecule.
You will be provided with example molecules and their continuous numerical values.

Please predict the numerical value for the given molecule.
Format your output exactly as a single number.
Example output: -3.14
"""


FreeSolv_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict the hydration free energy (in kcal/mol) of the molecule in water.
You will be provided with example molecules and their continuous numerical values.

Please predict the numerical value for the given molecule.
Format your output exactly as a single number.
Example output: -2.54
"""


Lipophilicity_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict the octanol/water distribution coefficient (logD at pH 7.4) of the molecule.
You will be provided with example molecules and their continuous numerical values.

Please predict the numerical value for the given molecule.
Format your output exactly as a single number.
Example output: 1.23
"""


BACE_r_prompt_N_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) binding affinity (pIC50) of the molecule.
You will be provided with example molecules and their continuous numerical values.

Please predict the numerical value for the given molecule.
Format your output exactly as a single number.
Example output: 6.54
"""
