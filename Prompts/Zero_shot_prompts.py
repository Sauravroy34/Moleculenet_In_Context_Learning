# -----------------------------------------------------------------------------
# Zero-Shot Prompts
# -----------------------------------------------------------------------------

BBBP_prompt_zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule has blood-brain barrier penetration capability.

Please predict the binary label for the given molecule: Yes (penetrative) or No (non-penetrative).
Provide a probability score between 0.0 and 1.0 indicating the likelihood of it being penetrative (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


BACE_prompt_zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule can inhibit Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1).

Please predict the binary label for the given molecule: Yes (inhibitor) or No (non-inhibitor).
Provide a probability score between 0.0 and 1.0 indicating the likelihood of it being an inhibitor (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


Tox21_prompt_zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule is toxic.

Please predict the binary label for the given molecule: Yes (toxic) or No (not toxic).
Provide a probability score between 0.0 and 1.0 indicating the likelihood of it being toxic (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


HIV_prompt_zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SELFIES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule can inhibit HIV replication.

Please predict the binary label for the given molecule: Yes (can inhibit) or No (cannot inhibit).
Provide a probability score between 0.0 and 1.0 indicating the likelihood of it being able to inhibit HIV replication (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


ClinTox_prompt_zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule is clinically-trial-toxic or FDA approved.

Please predict the binary label for the given molecule: Yes (toxic/failed) or No (not toxic/approved).
Provide a probability score between 0.0 and 1.0 indicating the likelihood of it being toxic (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


SIDER_prompt_zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict whether the molecule has specific adverse drug reactions or side effects.

Please predict the binary label for the given molecule: Yes (has side effect) or No (does not have side effect).
Provide a probability score between 0.0 and 1.0 indicating the likelihood of it having the side effect (Yes).
Format your output exactly as:
[Yes/No], [Probability]
Example output: Yes, 0.85
"""


# Regression Prompts (Zero-Shot)

ESOL_prompt_zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict the water solubility (log solubility in mols per litre) of the molecule.

Please predict the continuous numerical value for the given molecule.
Format your output exactly as a single number.
Example output: -3.14
"""


FreeSolv_prompt_zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict the hydration free energy (in kcal/mol) of the molecule in water.

Please predict the continuous numerical value for the given molecule.
Format your output exactly as a single number.
Example output: -2.54
"""


Lipophilicity_prompt_zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict the octanol/water distribution coefficient (logD at pH 7.4) of the molecule.

Please predict the continuous numerical value for the given molecule.
Format your output exactly as a single number.
Example output: 1.23
"""


BACE_r_prompt_Zero_shot = """You are an expert chemist. Your task is to predict the property of a molecule based on its SMILES string representation.
Do not provide any reasoning, explanation, or additional text. Strictly follow the required output format.

Task: Predict the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) binding affinity (pIC50) of the molecule.

Please predict the continuous numerical value for the given molecule.
Format your output exactly as a single number.
Example output: 6.54
"""