#!/usr/bin/env python3
"""
OpenAI API Inference Script

This script performs inference using OpenAI's API with the following features:
- Loads configuration from a YAML file
- Downloads and samples from a datasets dataset
- Saves results to a specified output directory
"""

import os
from pathlib import Path
import json
import re
import string
import argparse
from typing import Dict, Any, List
from datasets import load_dataset
import pandas as pd
import yaml
from dotenv import load_dotenv
from utils import (
    load_llm,
    prepare_batch_prompts,
)


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


class VLLMInference:
    def __init__(self, ds_config: DatasetConfig, model_name_or_path: str, max_tokens: int,
                 system_prompt: str, output_dir: str, prompt_length_limit=512, data_sample=500,
                 gpu_num: int = 2, max_samples=50):
        self.ds_config = ds_config
        self.model_name_or_path = model_name_or_path
        self.model_name = Path(model_name_or_path).stem
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.output_dir = output_dir
        self.prompt_length_limit = prompt_length_limit
        self.data_sample = data_sample
        self.max_samples = max_samples
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, 'inference_results.jsonl')
        self.llm, self.sampling_params = load_llm(
            model_name_or_path=model_name_or_path,
            gpu_num=gpu_num,
        )

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

    def generate_batch(self, prompts: List[str]) -> List[str]:
        outputs_t = self.llm.chat(prompts, self.sampling_params, use_tqdm=True)
        pred_lst = []
        for o_t in outputs_t:
            pred_lst.append(o_t.outputs[0].text)
        return pred_lst

    def save_results(self, results: List[dict]) -> None:
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def run(self) -> None:
        print(f"Loading dataset: {self.ds_config.dataset_path}")
        datasets = self.load_and_sample_dataset()

        prompts = prepare_batch_prompts(datasets, self.ds_config.prompt_template, self.system_prompt)
        infer_texts = self.generate_batch(prompts)
        results = []
        idx = 0
        for dataset, prompt, infer_text in zip(datasets, prompts, infer_texts):
            result = {
                'id': idx,
                'model': self.model_name,
                'prompt': prompt,
                'response': infer_text,
            } | dataset
            results.append(result)
            idx += 1
        self.save_results(results)


def main():
    parser = argparse.ArgumentParser(description="vLLM Batch Offline Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/writingbench.yaml",
        help="Path to dataset config yaml",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="/group_homes/our_llm_domain/home/share/open_models/Qwen/Qwen2.5-14B-Instruct",
        help="vLLM model name or path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt prefix",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/writingbench/Qwen2.5-14B-Instruct",
        help="Output directory",
    )
    parser.add_argument(
        "--gpu-num",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--gpu_mem_util",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype for vLLM: auto, float16, bfloat16, float32",
    )

    args = parser.parse_args()

    ds_config = DatasetConfig(args.config)

    inference = VLLMInference(
        ds_config=ds_config,
        model_name_or_path=args.model_name_or_path,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        output_dir=args.output_dir,
        gpu_num=args.gpu_num,
    )

    inference.run()


if __name__ == "__main__":
    load_dotenv(override=True)
    main()