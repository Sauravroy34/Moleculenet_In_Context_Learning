import json
import os
import time
from uuid import uuid4

import anthropic
from openai import OpenAI
import litellm
from langfuse import get_client, propagate_attributes
import pandas as pd
from dotenv import load_dotenv

from Prompts.prompts import build_messages, output_schema, system_text, task_keys

# Load environment variables from .env file
load_dotenv(override=True)

litellm.drop_params = True

# Optionally initialize Langfuse
if os.environ.get('LANGFUSE_PUBLIC_KEY') and os.environ.get('LANGFUSE_SECRET_KEY'):
    LANGFUSE = get_client()
else:
    LANGFUSE = None
    print('WARNING: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set - nothing will be logged to Langfuse')


def provider_of(model):
    if '/' in model:
        return model.split('/', 1)[0]
    return 'anthropic' if model.startswith('claude') else 'openai'

def bare_model(model):
    return model.split('/', 1)[1] if '/' in model else model

def max_tokens(model, dataset_tasks):
    # Base tokens + tasks output tokens + thinking headroom
    headroom = 4096 if "claude-3-7" in model or "o1" in model or "o3" in model else 1024
    return 32 + 12 * len(dataset_tasks) + headroom


def json_response_format(dataset_name):
    return {'type': 'json_schema',
            'json_schema': {'name': 'prediction', 'strict': True, 'schema': output_schema(dataset_name)}}


class ModelInterface:
    def __init__(self, model_name: str, max_retries: int = 3, **kwargs):
        self.model_name = model_name
        self.max_retries = max_retries
        self.kwargs = kwargs

    def generate(self, dataset_name: str, messages: list) -> str:
        raise NotImplementedError


class LiteLLMModel(ModelInterface):
    """Realtime synchronous generation wrapper using litellm with Langfuse tracing."""
    
    def generate(self, dataset_name: str, messages: list) -> str:
        provider = provider_of(self.model_name)
        model = bare_model(self.model_name)
        
        # Prepare kwargs
        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.kwargs.get("temperature", 0.0),
        }
            
        for attempt in range(self.max_retries):
            try:
                # langfuse litellm callback is auto-handled by litellm if configured,
                # but we can explicitly set callbacks.
                litellm.success_callback = ["langfuse"] if LANGFUSE else []
                litellm.failure_callback = ["langfuse"] if LANGFUSE else []
                
                response = litellm.completion(**completion_kwargs)
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                time.sleep(2)
        return ""


def submit_batch_openai(model_name: str, dataset_name: str, requests: list, batch_id: str, temperature: float = 0.0):
    """Submit a batch job using OpenAI's Batch API."""
    client = OpenAI()
    
    # 1. Create JSONL file for requests
    jsonl_path = f"batch_{batch_id}.jsonl"
    with open(jsonl_path, "w") as f:
        for idx, msgs in enumerate(requests):
            req = {
                "custom_id": f"{dataset_name}-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": bare_model(model_name),
                    "messages": msgs,
                    "temperature": temperature,
                }
            }
            f.write(json.dumps(req) + "\n")
            
    # 2. Upload file
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
        
    # 3. Create batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"OpenAI batch submitted. Batch ID: {batch.id}")
    return batch.id


def submit_batch_anthropic(model_name: str, dataset_name: str, requests: list, batch_id: str, temperature: float = 0.0):
    """Submit a batch job using Anthropic's Message Batches API."""
    client = anthropic.Anthropic()
    
    batch_requests = []
    for idx, msgs in enumerate(requests):
        system = [dict(p) for p in msgs[0]['content']]
        
        # Caching the system prompt
        system[-1]['cache_control'] = {'type': 'ephemeral'}
            
        req = {
            "custom_id": f"{dataset_name}-{idx}",
            "params": {
                "model": bare_model(model_name),
                "system": system,
                "messages": [msgs[1]],
                "max_tokens": max_tokens(model_name, task_keys(dataset_name)),
                "temperature": temperature
            }
        }
        batch_requests.append(req)
        
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"Anthropic batch submitted. Batch ID: {batch.id}")
    return batch.id

