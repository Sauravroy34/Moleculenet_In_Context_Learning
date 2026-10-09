import argparse
import json
import pandas as pd
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

from DataLoader.loader import load_datsets
from Prompts.Dataset_task_map import DATASETS
from Prompts.prompts import parse_response
from main import metric_eval

# Load API keys
load_dotenv(override=True)

def retrieve_openai_batch(batch_id):
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    print(f"Batch Status: {batch.status}")
    
    if batch.status == "completed":
        file_response = client.files.content(batch.output_file_id)
        results = []
        for line in file_response.text.strip().split('\n'):
            if not line: continue
            results.append(json.loads(line))
        return results
    elif batch.status in ["failed", "cancelled", "expired"]:
        print("Batch did not complete successfully.")
        return None
    else:
        print("Batch is still processing... Check back later.")
        return None


def retrieve_anthropic_batch(batch_id):
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch Status: {batch.processing_status}")
    
    if batch.processing_status == "ended":
        results = []
        for result in client.messages.batches.results(batch_id):
            results.append(result.model_dump())
        return results
    elif batch.processing_status in ["canceling", "canceled"]:
        print("Batch cancelled.")
        return None
    else:
        print("Batch is still processing... Check back later.")
        return None


def process_results(results, dataset_name, provider):
    dataset_info = DATASETS[dataset_name]
    task_type = dataset_info['type']
    
    # Reload the exact same test dataset to map back the true labels using the index
    train, valid, test = load_datsets(dataset_name)
    test_df = test.to_dataframe()
    test_df = test_df[test_df["w"].apply(lambda x: any(w > 0 for w in (x if hasattr(x, '__iter__') else [x])))]
    
    prediction_records = []
    
    for res in results:
        custom_id = res['custom_id']
        idx = int(custom_id.split('-')[-1])  # Extract the index we embedded during submission
        
        # Get true data from dataframe using the extracted index
        row = test_df.iloc[idx]
        input_smiles = row["ids"]
        true_labels = row["y"]
        weights = row["w"]
        
        if not hasattr(true_labels, '__iter__'):
            true_labels = [true_labels]
            weights = [weights]
            
        # Extract output text
        if provider == "openai":
            try:
                raw_output = res['response']['body']['choices'][0]['message']['content']
            except KeyError:
                raw_output = ""
        elif provider == "anthropic":
            try:
                # In Anthropic's result, success payload is in ['result']['message']['content']
                if res['result']['type'] == 'succeeded':
                    message = res['result']['message']
                    raw_output = next((block['text'] for block in message['content'] if block['type'] == 'text'), "")
                else:
                    raw_output = ""
            except KeyError:
                raw_output = ""
                
        # Parse the JSON response
        preds, error = parse_response(dataset_name, raw_output)
        
        task_keys = list(dataset_info['tasks'].keys())
        for i, task_name in enumerate(task_keys):
            if weights[i] > 0:
                prediction_records.append({
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "smiles": input_smiles,
                    "task_key": task_name,
                    "true_label": true_labels[i],
                    "predicted": preds.get(task_name),
                    "error": error
                })
                
    out_df = pd.DataFrame(prediction_records)
    metrics = metric_eval(task_type, out_df, dataset_name, f"Batch-{provider}")
    return out_df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and evaluate completed Batch jobs.")
    parser.add_argument("--batch_id", type=str, required=True, help="API Batch ID (e.g., batch_xyz123)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name e.g., tox21")
    parser.add_argument("--provider", type=str, choices=["openai", "anthropic"], required=True, help="Which provider you used")
    args = parser.parse_args()
    
    if args.provider == "openai":
        results = retrieve_openai_batch(args.batch_id)
    else:
        results = retrieve_anthropic_batch(args.batch_id)
        
    if results:
        print("Parsing results and calculating metrics...")
        df, metrics = process_results(results, args.dataset, args.provider)
        
        print("\n=== Batch Metrics ===")
        import pprint
        pprint.pprint(metrics)
        
        out_csv = f"{args.dataset}_{args.provider}_batch_results.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved detailed predictions to {out_csv}")
