import random
import time
import pandas as pd 
import deepchem as dc
from tqdm import tqdm
from litellm import completion
from DataLoader.loader import load_datsets
from Prompts.N_shot_prompts import BBBP_prompt_N_shot, BACE_prompt_N_shot, Tox21_prompt_N_shot, HIV_prompt_N_shot, ClinTox_prompt_N_shot, SIDER_prompt_N_shot, ESOL_prompt_N_shot, FreeSolv_prompt_N_shot, Lipophilicity_prompt_N_shot, BACE_r_prompt_N_shot
from Prompts.Zero_shot_prompts import BBBP_prompt_Zero_shot, BACE_prompt_Zero_shot, Tox21_prompt_Zero_shot, HIV_prompt_Zero_shot, ClinTox_prompt_Zero_shot, SIDER_prompt_Zero_shot, ESOL_prompt_Zero_shot, FreeSolv_prompt_Zero_shot, Lipophilicity_prompt_Zero_shot, BACE_r_prompt_Zero_shot
from sklearn.metrcis import roc_auc_score,accuracy_score,root_mean_squared_error

SEED = 42

Model_Name = " "

dataset_name = "bace_c"

train , valid , test = load_datsets(dataset_name)

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
    "regression": ["bace_r", "esol" , "freesolv" , "lipophilicity"]
}

def generate_response(prompt, n=1, max_retries=6,temperature = 0.7):
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
            
                    
        outputs.append(text)
        time.sleep(0.5)                         
    return outputs



def random_sample_examples(dataset,sample_size, task, SEED):

    random.seed(SEED)
    df = dataset.to_dataframe()

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


def metric_eval(task,df):

    if task == "classification":
        confidences_yes = df["confidence_yes"].to_list()
        labels = df["true_label"].apply(lambda x: 1 if x.lower() == "yes" else 0).to_list()

        auc = roc_auc_score(labels, confidences_yes)
       
        prediction = df['predicted'].apply(lambda x: 1 if x.lower() == "yes" else 0).to_list()
        true = labels

        accuracy = accuracy_score(true,prediction)

        f1_score = f1_score(true,prediction)

        return {"auc": auc,"accuracy": accuracy,"f1_score":f1_score}


    if task == "regression":
        predicted = df["predicted"].tolist()
        true = df["true_label"].tolist()

        rmse = root_mean_squared_error(true,predicted)
        

        return {"rmse": rmse}



def single_run(name,k,SEED):

    task = ""

    train , valid , test = load_datsets(dataset_name)

    for task,datasets in TASK_MAP.items():
        if name in datasets:
            task = task 
        else:
            raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(ZERO_SHOT_MAP.keys())}")


    if task == "classification":
        columns = ["name","k","task",'smiles', 'true_label ', 'Predicted','confidence','confidence_yes']
    
    else:
        columns = ["name","k","task",'smiles', 'true_label ', 'Predicted']

    
    prediction = []

    
        
    for i in tqdm(range(len(test))):
        test_df = test.to_dataframe()
        test_df = test_df[test_df["w"] > 0]

        input_smiles = test_df.iloc[i]["ids"]
        true_label = test_df.iloc[i]["y"]

        if task == "classification":
            if true_label == 1:
                true_label = "Yes"
            else:
                label = "No"
        

        if k == 0:
            prompts = create_prompt(input_smiles = input_smiles, name=name,examples= None)


        else:
            examples = random_sample_examples(dataset = train, sample_size=k, task=task, SEED=SEED)
            prompts = create_prompt(input_smiles = input_smiles,name = name,examples = examples)

    
        out = generate_response(prompts,n = 1)

        if task == "classification":
            to_append = [name,k,task,input_smiles,true_label,out[0][0] , out[0][1], (1 -out[0][1] if out[0][0].lower() == "no" else out[0][1])]
        
        else:
            to_append = [name,k,task,input_smiles,true_label,out[0][0]]

        prediction.append(to_append)

    out_df = pd.DataFrame(prediction,columns=columns)

    metrics = metric_eval(task,out_df)

    return out_df , metrics
    



    





    





    

        
        
