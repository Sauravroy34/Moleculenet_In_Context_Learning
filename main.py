import argparse
import random
import os
import time
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score

from DataLoader.loader import load_datsets
from Prompts.Dataset_task_map import DATASETS
from Prompts.prompts import build_messages, parse_response
from backends import LiteLLMModel, provider_of, submit_batch_openai, submit_batch_anthropic

SEED = 42

def metric_eval(task_type, df, dataset_name, model_name):
    """
    Evaluate metrics for multi-task predictions.
    df contains columns: ['task_key', 'true_label', 'predicted', ...]
    We calculate per-task metrics and average them.
    """
    metrics = {"model_name": model_name, "dataset": dataset_name}
    
    if task_type == "classification":
        auc_list = []
        acc_list = []
        f1_list = []
        
        for task_key in df['task_key'].unique():
            task_df = df[df['task_key'] == task_key].dropna(subset=['true_label'])
            if len(task_df) == 0:
                continue
                
            y_true = task_df['true_label'].astype(float).tolist()
            # Predicted is probability for classification
            y_pred_prob = task_df['predicted'].fillna(0.0).astype(float).tolist()
            y_pred_class = [1 if p >= 0.5 else 0 for p in y_pred_prob]
            
            try:
                auc = roc_auc_score(y_true, y_pred_prob)
                auc_list.append(auc)
            except ValueError:
                pass # Only one class present in true labels
                
            acc_list.append(accuracy_score(y_true, y_pred_class))
            f1_list.append(f1_score(y_true, y_pred_class))
            
        metrics["roc_auc"] = sum(auc_list) / len(auc_list) if auc_list else 0.0
        metrics["accuracy"] = sum(acc_list) / len(acc_list) if acc_list else 0.0
        metrics["f1_score"] = sum(f1_list) / len(f1_list) if f1_list else 0.0
        
    elif task_type == "regression":
        rmse_list = []
        for task_key in df['task_key'].unique():
            task_df = df[df['task_key'] == task_key].dropna(subset=['true_label'])
            if len(task_df) == 0:
                continue
                
            y_true = task_df['true_label'].astype(float).tolist()
            y_pred = task_df['predicted'].fillna(0.0).astype(float).tolist()
            
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            rmse_list.append(rmse)
            
        metrics["rmse"] = sum(rmse_list) / len(rmse_list) if rmse_list else 0.0
        
    return metrics


def single_run(name: str, k: int, SEED: int, model_name: str, mode: str = "realtime", limit: int = None):
    """
    Run evaluation for a dataset using the specified model.
    mode can be 'realtime' or 'batch'.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(DATASETS.keys())}")
        
    dataset_info = DATASETS[name]
    task_type = dataset_info['type']
    
    train, valid, test = load_datsets(name)
    test_df = test.to_dataframe()
    # Filter rows with at least one task weight > 0
    test_df = test_df[test_df["w"].apply(lambda x: any(w > 0 for w in (x if hasattr(x, '__iter__') else [x])))]
    
    if limit:
        test_df = test_df.head(limit)
        
    model = LiteLLMModel(model_name=model_name)
    
    prediction_records = []
    batch_requests = []
    
    print(f"Starting {mode} run for {name} with k={k} using {model_name}...")
    
    for i in tqdm(range(len(test_df))):
        input_smiles = test_df.iloc[i]["ids"]
        true_labels = test_df.iloc[i]["y"]
        weights = test_df.iloc[i]["w"]
        
        if not hasattr(true_labels, '__iter__'):
            true_labels = [true_labels]
            weights = [weights]
            
        messages = build_messages(name, train, input_smiles, k, SEED)
        
        if mode == "batch":
            batch_requests.append(messages)
            # Save truths for later
            prediction_records.append({
                "smiles": input_smiles,
                "true_labels": true_labels,
                "weights": weights
            })
        else:
            # Realtime execution
            raw_output = model.generate(name, messages)
            preds, error = parse_response(name, raw_output)
            
            task_keys = list(dataset_info['tasks'].keys())
            for idx, task_name in enumerate(task_keys):
                if weights[idx] > 0: # Only record if measured
                    prediction_records.append({
                        "dataset": name,
                        "k": k,
                        "task_type": task_type,
                        "smiles": input_smiles,
                        "task_key": task_name,
                        "true_label": true_labels[idx],
                        "predicted": preds.get(task_name),
                        "error": error
                    })
                    
    if mode == "batch":
        batch_id = f"batch_{name}_k{k}_{int(time.time())}"
        provider = provider_of(model_name)
        if provider == "openai":
            api_batch_id = submit_batch_openai(model_name, name, batch_requests, batch_id)
        elif provider == "anthropic":
            api_batch_id = submit_batch_anthropic(model_name, name, batch_requests, batch_id)
        else:
            raise ValueError(f"Batch mode not supported for provider {provider}")
            
        print(f"Batch submitted successfully! API Batch ID: {api_batch_id}")
        return pd.DataFrame(), {}
        
    else:
        out_df = pd.DataFrame(prediction_records)
        metrics = metric_eval(task_type, out_df, name, model_name)
        return out_df, metrics


if __name__ == "__main__":
    # Example test
    # df, metrics = single_run("bbbp", k=5, SEED=SEED, model_name="gpt-4o", mode="realtime", limit=3)
    # print(metrics)
    pass
