from ast import Pass
import random
import time
import pandas as pd 
import deepchem as dc
from tqdm import tqdm
from litellm import completion
from DataLoader.loader import load_datsets
from Prompts.N_shot_prompts import BBBP_prompt_N_shot, BACE_prompt_N_shot, Tox21_prompt_N_shot, HIV_prompt_N_shot, ClinTox_prompt_N_shot, SIDER_prompt_N_shot, ESOL_prompt_N_shot, FreeSolv_prompt_N_shot, Lipophilicity_prompt_N_shot, BACE_r_prompt_N_shot
from Prompts.Zero_shot_prompts import BBBP_prompt_Zero_shot, BACE_prompt_Zero_shot, Tox21_prompt_Zero_shot, HIV_prompt_Zero_shot, ClinTox_prompt_Zero_shot, SIDER_prompt_Zero_shot, ESOL_prompt_Zero_shot, FreeSolv_prompt_Zero_shot, Lipophilicity_prompt_Zero_shot, BACE_r_prompt_Zero_shot
from sklearn.metrics import roc_auc_score, accuracy_score, root_mean_squared_error, f1_score

SEED = 42

N_SHOT_MAP = {
    "bace_c" : BACE_prompt_N_shot,
    "bace_r" : BACE_r_prompt_N_shot,
    "bbbp": BBBP_prompt_N_shot,
    "tox21": Tox21_prompt_N_shot,
    "hiv": HIV_prompt_N_shot,
    "clintox": ClinTox_prompt_N_shot,
    "sider": SIDER_prompt_N_shot,
    "esol": ESOL_prompt_N_shot,
    "freesolv": FreeSolv_prompt_N_shot,
    "lipo": Lipophilicity_prompt_N_shot
}

ZERO_SHOT_MAP = {
    "bace_c" : BACE_prompt_Zero_shot,
    "bace_r" : BACE_r_prompt_Zero_shot,
    "bbbp": BBBP_prompt_Zero_shot,
    "tox21": Tox21_prompt_Zero_shot,
    "hiv": HIV_prompt_Zero_shot,
    "clintox": ClinTox_prompt_Zero_shot,
    "sider": SIDER_prompt_Zero_shot,
    "esol": ESOL_prompt_Zero_shot,
    "freesolv": FreeSolv_prompt_Zero_shot,
    "lipo": Lipophilicity_prompt_Zero_shot
}

TASK_MAP = {
    "classification": ["bace_c" , "bbbp" , "tox21" , "hiv" , "clintox" , "sider"],
    "regression": ["bace_r", "esol" , "freesolv" , "lipo"]
}




class ModelInterface:

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        raise NotImplementedError

        




class LiteLLMModel(ModelInterface):
    """
    Modular wrapper for models supported by litellm 
    (OpenAI, Anthropic, HuggingFace, Cohere, vLLM, etc.)
    
    Usage examples:
        model = LiteLLMModel(model_name="gpt-4")
        model = LiteLLMModel(model_name="claude-3-opus-20240229")
        model = LiteLLMModel(model_name="huggingface/meta-llama/Llama-2-7b-chat-hf")
        model = LiteLLMModel(model_name="openai/custom-model", api_base="http://localhost:8000/v1")
    """
    def __init__(self, model_name: str, api_key: str = None, api_base: str = None, max_retries: int = 6, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self.kwargs = kwargs

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        outputs = []
        for _ in range(n):
            text = ""
            for attempt in range(self.max_retries):
                try:
                    response = completion(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        api_key=self.api_key,
                        api_base=self.api_base,
                        **self.kwargs
                    )
                    text = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                    time.sleep(2) # Exponential backoff can be added here
            outputs.append(text)
            time.sleep(0.5)                         
        return outputs


# =====================================================================
# Core Pipeline
# =====================================================================

def random_sample_examples(dataset, sample_size, task, SEED):
    random.seed(SEED)
    df = dataset.to_dataframe()

    if task == "classification":
        positive = df[(df["y"] == 1) & (df["w"] > 0)].sample(int(sample_size // 2), random_state=SEED)
        negative = df[(df["y"] == 0) & (df["w"] > 0)].sample(int(sample_size // 2), random_state=SEED)

        smiles = positive["ids"].tolist() + negative["ids"].tolist()
        class_label = positive["y"].tolist() + negative["y"].tolist()
        class_label = ["Yes" if i == 1 else "No" for i in class_label]

        examples = list(zip(smiles, class_label))
        return examples

    if task == "regression":
        examples = df[df["w"] > 0].sample(sample_size, random_state=SEED)
        smiles = examples["ids"].tolist()
        y = examples["y"].tolist()
        examples = list(zip(smiles, y))
        return examples


def create_prompt(input_smiles, name, examples=None):
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


def metric_eval(task,df):

    if task == "classification":
        confidences_yes = df["confidence_yes"].to_list()
        labels = df["true_label"].apply(lambda x: 1 if x.lower() == "yes" else 0).to_list()

        lables = [float(i) for i in labels]
        confidences_yes = [float(i) for i in confidences_yes]

        labels_ = labels * 100
        confidences_yes_ = confidences_yes * 100
        auc = roc_auc_score(labels_, confidences_yes_)
       
        prediction = df['Predicted'].apply(lambda x: 1 if x.lower() == "yes" else 0).to_list()
        true = labels

        accuracy = accuracy_score(true,prediction)

        f1 = f1_score(true,prediction)

        return {"auc": auc,"accuracy": accuracy,"f1_score":f1}


    if task == "regression":
        predicted = df["Predicted"].tolist()
        true = df["true_label"].tolist()
        
        predicted = [float(i) for i in predicted]
        true = [float(i) for i in true]

        rmse = root_mean_squared_error(true,predicted)
        

        return {"rmse": rmse}



def single_run(name: str, k: int, SEED: int, model: ModelInterface):
    task = None
    for task_name, datasets in TASK_MAP.items():
        if name in datasets:
            task = task_name 
            break
            
    if task is None:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(ZERO_SHOT_MAP.keys())}")

    train, valid, test = load_datsets(name)

    if task == "classification":
        columns = ["name", "k", "task", "smiles", "true_label", "Predicted", "confidence", "confidence_yes"]
    else:
        columns = ["name", "k", "task", "smiles", "true_label", "Predicted"]

    prediction = []
    
    test_df = test.to_dataframe()
    test_df = test_df[test_df["w"] > 0]
        
    for i in tqdm(range(len(test_df))):
        input_smiles = test_df.iloc[i]["ids"]
        true_label = test_df.iloc[i]["y"]

        if task == "classification":
            if true_label == 1:
                true_label = "Yes"
            else:
                true_label = "No"

        if k == 0:
            prompts = create_prompt(input_smiles=input_smiles, name=name, examples=None)
        else:
            examples = random_sample_examples(dataset=train, sample_size=k, task=task, SEED=SEED)
            prompts = create_prompt(input_smiles=input_smiles, name=name, examples=examples)

        out = model.generate(prompts, n=1)
        raw_output = out[0]

        if task == "classification":
            # Attempt to parse output formatted as [Yes/No], [Probability]
            try:
                parts = [p.strip() for p in raw_output.split(",")]
                pred_label = parts[0]
                conf = float(parts[1]) if len(parts) > 1 else 0.0
                conf_yes = conf if pred_label.lower() == "yes" else 1.0 - conf
            except Exception:
                pred_label = "Error"
                conf = 0.0
                conf_yes = 0.0
                
            to_append = [name, k, task, input_smiles, true_label, pred_label, conf, conf_yes]
            
        else:
            # Parse numerical output for regression
            try:
                pred_val = float(raw_output)
            except Exception:
                pred_val = 0.0
            to_append = [name, k, task, input_smiles, true_label, pred_val]

        prediction.append(to_append)

    out_df = pd.DataFrame(prediction, columns=columns)
    metrics = metric_eval(task, out_df)

    return out_df, metrics
