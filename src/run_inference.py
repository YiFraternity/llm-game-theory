#!/usr/bin/env python3
"""
OpenAI API Inference Script

This script performs inference using OpenAI's API with the following features:
- Loads configuration from a YAML file
- Downloads and samples from a datasets dataset
- Saves results to a specified output directory
- Uses OpenAI API v1.0+
"""

import os
import asyncio
import json
import argparse
from typing import Dict, Any, List, Set
import logging
from datasets import load_dataset
import pandas as pd
import yaml
from openai import AsyncOpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from utils import (
    populate_template,
    async_retry_on_api_error,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class DatasetConfig:
    def __init__(self, config_path: str):
        """Initialize dataset configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dataset_path: str = self.config.get('dataset_path')
        self.dataset_split: str = self.config.get('dataset_split', 'test')
        self.prompt_template: str = self.config.get('prompt_template', '{text}')

class OpenAIInference:
    def __init__(self, ds_config: DatasetConfig, model: str, max_tokens: int,
                 temperature: float, system_prompt: str, output_dir: str,
                 MAX_IN_FLIGHT_REQUESTS=10):
        self.client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE'),
            timeout=1200,
            max_retries=0,
        )
        self.ds_config = ds_config
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.output_dir = output_dir
        self.prompt_length_limit = 512
        self.data_sample = 500
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, 'inference_results.jsonl')
        self._semaphore = asyncio.Semaphore(MAX_IN_FLIGHT_REQUESTS)
        self._file_lock = asyncio.Lock()
        logger.info(f'api-base: {os.getenv("OPENAI_API_BASE")}; model-name: {self.model}')

    def load_and_sample_dataset(self) -> List[Dict[str, Any]]:
        try:
            # 判断是否为本地文件
            local_exts = ['.jsonl', '.json', '.csv', '.tsv', '.xlsx']
            path = self.ds_config.dataset_path.strip()
            is_local = any(path.endswith(ext) for ext in local_exts)
            if is_local:
                if path.endswith('.jsonl'):
                    df = pd.read_json(path, lines=True)
                elif path.endswith('.json'):
                    df = pd.read_json(path)
                elif path.endswith('.csv'):
                    df = pd.read_csv(path)
                elif path.endswith('.tsv'):
                    df = pd.read_csv(path, sep='\t')
                elif path.endswith('.xlsx'):
                    df = pd.read_excel(path)
                else:
                    raise ValueError(f"Unsupported local file format: {path}")
                dataset = df.to_dict(orient='records')
            else:
                dataset = load_dataset(path, split=self.ds_config.dataset_split)
            if len(dataset) > self.data_sample:
                dataset = dataset[:self.data_sample]
            return dataset
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}") from e

    @async_retry_on_api_error(max_attempts=3, min_wait=30, max_wait=240)
    async def generate_response(self, prompt: str) -> str:
        async with self._semaphore:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        return resp.choices[0].message.content.strip()

    async def save_result_append(self, result: dict) -> None:
        async with self._file_lock:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def load_finished_ids(self) -> Set[int]:
        finished = set()
        if not os.path.exists(self.output_file):
            return finished

        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    finished.add(obj["id"])
                except Exception:
                    continue
        return finished

    async def append_result(self, result: Dict[str, Any]) -> None:
        async with self._file_lock:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    async def run(self) -> None:
        print(f"Loading dataset: {self.ds_config.dataset_path}")
        dataset = self.load_and_sample_dataset()

        finished_ids = self.load_finished_ids()
        print(f"Resume: skip {len(finished_ids)} finished samples")

        async def worker(i: int, example: Dict[str, Any]):
            prompt = populate_template(self.ds_config.prompt_template, example)
            response = await self.generate_response(prompt)
            result = {
                "id": i,
                "model": self.model,
                "prompt": prompt,
                "response": response,
                **example,
            }
            await self.append_result(result)

        tasks = [
            worker(i, ex)
            for i, ex in enumerate(dataset)
            if i not in finished_ids
        ]

        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Processing",
        ):
            await coro

        print(f"Inference complete. Results saved to {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI API Inference")
    parser.add_argument('--config', type=str, default='configs/writingbench-ranks.yaml', help='Path to dataset config yaml')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-05-13', help='OpenAI model name')
    parser.add_argument('--max-tokens', type=int, default=64, help='Max tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--system-prompt', type=str, default='You are a helpful AI assistant.', help='System prompt')
    parser.add_argument('--output-dir', type=str, default='./outputs/writingbench/ranks/gpt-4o-2024-05-13', help='Output directory')
    args = parser.parse_args()

    ds_config = DatasetConfig(args.config)
    inference = OpenAIInference(
        ds_config,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        output_dir=args.output_dir
    )
    asyncio.run(inference.run())

if __name__ == "__main__":
    load_dotenv(override=True)
    main()

