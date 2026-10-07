import random
import time
import pandas as pd 
import deepchem as dc
from litellm import completion
from DataLoader.loader import load_datsets
from Prompts.N_shot_prompts import BBBP_prompt_N_shot, BACE_prompt_N_shot, Tox21_prompt_N_shot, HIV_prompt_N_shot, ClinTox_prompt_N_shot, SIDER_prompt_N_shot, ESOL_prompt_N_shot, FreeSolv_prompt_N_shot, Lipophilicity_prompt_N_shot
from Prompts.Zero_shot_prompts import BBBP_prompt_Zero_shot, BACE_prompt_Zero_shot, Tox21_prompt_Zero_shot, HIV_prompt_Zero_shot, ClinTox_prompt_Zero_shot, SIDER_prompt_Zero_shot, ESOL_prompt_Zero_shot, FreeSolv_prompt_Zero_shot, Lipophilicity_prompt_Zero_shot


SEED = 42

Model_Name = " "

dataset_name = "bace_c"

train , valid , test = load_datsets(dataset_name)

random.seed(42)
N_SHOT_MAP = {
    "bace_c" : BACE_prompt_N_shot,
    "bbbp": BBBP_prompt_N_shot,
    "tox21": Tox21_prompt_N_shot,
    "hiv": HIV_prompt_N_shot,
    "clintox": ClinTox_prompt_N_shot,
    "sider": SIDER_prompt_N_shot,
    "esol": ESOL_prompt_N_shot,
    "freesolv": FreeSolv_prompt_N_shot,
    "lipophilicity": Lipophilicity_prompt_N_shot
}



ZERO_SHOT_MAP = {
    "bace_c" : BACE_prompt_Zero_shot,
    "bbbp": BBBP_prompt_Zero_shot,
    "tox21": Tox21_prompt_Zero_shot,
    "hiv": HIV_prompt_Zero_shot,
    "clintox": ClinTox_prompt_Zero_shot,
    "sider": SIDER_prompt_Zero_shot,
    "esol": ESOL_prompt_Zero_shot,
    "freesolv": FreeSolv_prompt_Zero_shot,
    "lipophilicity": Lipophilicity_prompt_Zero_shot
}

def generate_response(prompt, n=5, max_retries=6,temperature = 0.7):
    outputs = []
    for _ in range(n):
        text = ""
        for attempt in range(max_retries):
            try:
                response = completion(
                    model=Model_Name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                print("API error:", e)
                break
                    
        outputs.append(text)
        time.sleep(0.5)                         
    return outputs



def random_sample_examples(sample_size, task):
    df = train.to_dataframe()

    if task == "classification":
        positive = df[df["y"] == 1 and df["w"] > 0].sample(int(sample_size//2))
        negative = df[df["y"] == 0 and df["w"] > 0].sample(int(sample_size//2))

        smiles = positive["ids"].tolist() + negative["ids"].tolist()

        class_label = positive["y"].tolist() + negative["y"].tolist()

        class_label = ["Yes" if i == 1 else "No" for i in class_label]

        examples = list(zip(smiles,class_label))

        return examples

    if task == "regression":
        examples = df[df["w"] > 0].sample(sample_size)

        smiles = examples["ids"].tolist()
        y = examples["y"].tolist()

        examples = list(zip(smiles,y))

        return examples
        

def create_prompt(input_smiles, name, examples = None):

    if examples is None:
        prompt = ZERO_SHOT_MAP[name]
        prompt += f"\nInput smiles: {input_smiles}\n"
        return prompt
    else:
        prompt = N_SHOT_MAP[name]
        
        for example in examples:
            prompt += f"SMILES: {example[0]}\n"
            prompt += f"Property: {example[-1]}\n"
        prompt += f"SMILES: {input_smiles}\n"
        return prompt



examples = random_sample_examples(10,"classification") 

input_smiles = "CCCCCCCCCC"
prompt = create_prompt(input_smiles,dataset_name,examples)


out_put = generate_response(prompt)
