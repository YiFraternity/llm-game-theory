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
import json
import re
import string
import argparse
from typing import Dict, Any, List
from datasets import load_dataset
import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from jinja2 import Template, StrictUndefined
from utils import count_text_components


def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


def get_paragraph_length(text: str) -> int:
    """
    Calculate the length of a paragraph in terms of characters,
    including Chinese characters, Chinese punctuation, English words, and English punctuation.
    """
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    chinese_punct_pattern = re.compile(r'[，。！？；：（）《》、“”‘’]')
    english_word_pattern = re.compile(r'\b[a-zA-Z]+\b')
    english_punct_pattern = re.compile(r'[{}]'.format(re.escape(string.punctuation)))

    chinese_chars = chinese_char_pattern.findall(text)
    chinese_puncts = chinese_punct_pattern.findall(text)
    english_words = english_word_pattern.findall(text)
    english_puncts = english_punct_pattern.findall(text)

    return len(chinese_chars) + len(chinese_puncts) + len(english_words) + len(english_puncts)


class DatasetConfig:
    def __init__(self, config_path: str):
        """Initialize dataset configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dataset_path: str = self.config.get('dataset_path')
        self.dataset_split: str = self.config.get('dataset_split', 'test')
        self.prompt_template: str = self.config.get('prompt_template', '{text}')
        self.max_samples: int = self.config.get('max_samples', 100)


class OpenAIInference:
    def __init__(self, ds_config: DatasetConfig, model: str, max_tokens: int, temperature: float, system_prompt: str, output_dir: str):
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE'),
        )
        self.ds_config = ds_config
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.output_dir = output_dir
        self.prompt_length_limit = 512
        self.data_sample = 500
        self.max_samples = 50
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, 'inference_results.jsonl')

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

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return ""

    def save_result_append(self, result: dict, index: int) -> None:
        """Append a single result to the JSONL file."""
        if os.path.exists(self.output_file) and index != -1:
            data = []
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            data[index] = result
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            return
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def load_finished_prompts(self) -> set:
        """Load finished prompts from output file if exists."""
        finished_prompts, responses = [], []
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        finished_prompts.append(data.get('prompt'))
                        responses.append(data.get('response'))
                    except Exception:
                        continue
        return finished_prompts, responses

    def run(self) -> None:
        print(f"Loading dataset: {self.ds_config.dataset_path}")
        dataset = self.load_and_sample_dataset()
        finished_prompts, responses = self.load_finished_prompts()
        print(f"Running inference on {len(dataset)} samples... (skip {len(finished_prompts)} already done)")
        num = 0
        test_index = -1
        for i, example in enumerate(tqdm(dataset, desc="Processing")):
            prompt = populate_template(self.ds_config.prompt_template, example)
            test = example.get('prompt', prompt)
            try:
                test_index = finished_prompts.index(test)
            except ValueError:
                try:
                    test_index = finished_prompts.index(prompt)
                except ValueError:
                    test_index = -1
            # if (test_index != -1 and responses[test_index] != '') or num >= self.max_samples or count_text_components(prompt) > 512:
            if (test_index != -1 and responses[test_index] != '') or num >= self.max_samples:
                if test_index != -1:
                    num += 1
                continue
            response = self.generate_response(prompt)
            result = {
                'id': i,
                'model': self.model,
                'prompt': prompt,
                'response': response,
            } | example
            self.save_result_append(result, test_index)
            num += 1
        print(f"Inference complete! Results saved to {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI API Inference")
    parser.add_argument('--config', type=str, default='configs/writingbench-ranks.yaml', help='Path to dataset config yaml')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-05-13', help='OpenAI model name')
    parser.add_argument('--max_tokens', type=int, default=64, help='Max tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--system_prompt', type=str, default='You are a helpful AI assistant.', help='System prompt')
    parser.add_argument('--output_dir', type=str, default='./outputs/writingbench/ranks/gpt-4o-2024-05-13', help='Output directory')
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
    inference.run()

if __name__ == "__main__":
    load_dotenv(override=True)
    main()

