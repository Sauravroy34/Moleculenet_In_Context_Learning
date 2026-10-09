import json
import math
import re
from functools import cache
import numpy as np
import pandas as pd

from .Dataset_task_map import DATASETS

CLASSIFICATION_UNIT = 'probability between 0 and 1 (unitless) that the statement is true'


def task_keys(dataset):
    """Short JSON keys (t1..tN) used in place of long column names to save output tokens."""
    return [f't{i + 1}' for i in range(len(DATASETS[dataset]['tasks']))]

def output_schema(dataset):
    """Strict JSON schema of the answer: one required number per task key."""
    keys = task_keys(dataset)
    return {
        'type': 'object',
        'properties': {k: {'type': 'number'} for k in keys},
        'required': keys,
        'additionalProperties': False,
    }

def select_examples(dc_dataset, task_type, k, seed):
    """k random train examples, the same for every query of a dataset.
    dc_dataset is a deepchem Dataset object.
    """
    if k == 0:
        return []
        
    rng = np.random.default_rng(seed)
    y = dc_dataset.y
    w = dc_dataset.w
    ids = dc_dataset.ids
    
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
        w = w.reshape(-1, 1)

    if task_type == 'classification' and y.shape[1] == 1:
        y_flat = y[:, 0]
        w_flat = w[:, 0]
        
        pos_idx = np.where((y_flat == 1) & (w_flat > 0))[0]
        neg_idx = np.where((y_flat == 0) & (w_flat > 0))[0]
        
        pos_idx = rng.permutation(pos_idx)
        neg_idx = rng.permutation(neg_idx)
        
        n_pos = min(len(pos_idx), k // 2)
        idx = rng.permutation(np.concatenate([pos_idx[:n_pos], neg_idx[:k - n_pos]]))
    else:
        # Multitask or regression
        has_any_weight = np.where(np.any(w > 0, axis=1))[0]
        idx = rng.permutation(has_any_weight)[:k]
        
    examples = []
    for i in idx:
        examples.append({
            "smiles": ids[i],
            "y": y[i],
            "w": w[i]
        })
    return examples


def build_messages(dataset_name, dc_dataset, input_smiles, k, seed):
    """OpenAI-format messages asking for the properties of test molecule, with k examples."""
    ds = DATASETS[dataset_name]
    keys = task_keys(dataset_name)
    is_cls = ds['type'] == 'classification'

    lines = [
        'You are an expert medicinal and computational chemist. You predict molecular '
        'properties from SMILES strings.',
        '',
        f"Dataset: {ds['description']}",
        '',
        'For the molecule given by the user, predict:',
        *(f'- {key}: {desc}' for key, desc in zip(keys, ds['tasks'].values())),
        '',
        f'Units: every value is the {CLASSIFICATION_UNIT}.' if is_cls else f"Units: {ds.get('unit', '')}.",
        'Use chain-of-thought reasoning to analyze the target molecule, and enclose your detailed thinking process within <think></think> tags.',
        f'After thinking, respond with a JSON object with the keys {", ".join(keys)} and numeric values, '
        f'for example {json.dumps({key: 0.5 if is_cls else 0.0 for key in keys[:2]})}.'
    ]
    system_parts = [{'type': 'text', 'text': '\n'.join(lines)}]

    if dc_dataset is not None:
        examples = select_examples(dc_dataset, ds['type'], k, seed)
        if len(examples):
            header = 'Examples with measured values from the training set'
            header += ' (1 = true, 0 = false, null = not measured):' if is_cls else ':'
            blocks = [header]
            for ex in examples:
                labels = {}
                for i, key in enumerate(keys):
                    if ex['w'][i] == 0:
                        labels[key] = None
                    else:
                        labels[key] = int(ex['y'][i]) if is_cls else round(float(ex['y'][i]), 3)
                blocks.append(f"SMILES: {ex['smiles']}\nAnswer: {json.dumps(labels)}")
            system_parts.append({'type': 'text', 'text': '\n\n'.join(blocks)})

    # Enable prompt caching for Anthropic models on the largest static part (the system prompt)
    system_parts[-1]['cache_control'] = {"type": "ephemeral"}

    return [
        {'role': 'system', 'content': system_parts},
        {'role': 'user', 'content': f'SMILES: {input_smiles}'},
    ]


def system_text(messages):
    """The system message's text parts joined into one string."""
    return '\n\n'.join(p['text'] for p in messages[0]['content'])


_JSON_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)


def parse_response(dataset_name, text):
    """Map a model's JSON answer back to ({column: float or None}, error or None)."""
    cols = list(DATASETS[dataset_name]['tasks'])
    empty = {c: None for c in cols}
    if not text:
        return empty, 'empty response'
        
    # Remove <think>...</think> block if present to avoid parsing JSON inside it
    text_no_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Find the largest {...} block
    start_idx = text_no_think.find('{')
    end_idx = text_no_think.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return empty, 'no JSON object in response'
        
    json_str = text_no_think[start_idx:end_idx+1]
    
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        return empty, f'invalid JSON: {e}'

    is_cls = DATASETS[dataset_name]['type'] == 'classification'
    preds, missing = {}, []
    for key, col in zip(task_keys(dataset_name), cols):
        try:
            v = float(obj.get(key))
        except (TypeError, ValueError):
            v = math.nan
        if math.isnan(v) or math.isinf(v):
            preds[col] = None
            missing.append(key)
        else:
            preds[col] = min(max(v, 0.0), 1.0) if is_cls else v
    return preds, (f'missing keys: {missing}' if missing else None)
