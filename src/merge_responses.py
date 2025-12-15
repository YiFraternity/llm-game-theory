#!/usr/bin/env python3
"""
Merge inference results from multiple models into a single file.
Each entry contains all model responses for a given sample ID.

Expected directory structure:
outputs/
  ceval/
    model1/
      inference_results.jsonl
    model2/
      inference_results.jsonl
    ...
"""

import os
import argparse
import json
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List
from utils import (
    CHINESE_RANKING,
    INSTRUCTION_RANKING,
    CREATING_WRITING,
    CODEING,
)


def find_result_files(dataset_dir: str) -> List[str]:
    """Find all inference result files in the dataset directory."""
    pattern = os.path.join(dataset_dir, '*', 'inference_results.jsonl')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No result files found in {dataset_dir}. "
            f"Expected structure: {dataset_dir}/*/inference_results.jsonl"
        )
    return files


def load_results(dataset_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Load results from all model directories in the dataset directory.

    Args:
        dataset_dir: Path to the dataset directory (e.g., 'outputs/ceval')

    Returns:
        Dictionary mapping sample IDs to model responses
    """
    results = defaultdict(dict)
    files = find_result_files(dataset_dir)

    print(f"Found {len(files)} model result files in {dataset_dir}")

    for file_path in files:
        model_name = Path(file_path).parent.name
        print(f"Processing: {model_name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        sample_id = data.get('id')
                        if sample_id is not None:
                            results[sample_id][model_name] = {
                                'response': data.get('response', ''),
                                'prompt': data.get('text') or data.get('question') or data.get('query') or data.get('prompt'),
                            }
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line in {file_path}: {e}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return dict(results)

def save_merged_results(
    results: Dict[int, Dict[str, Any]],
    output_file: str,
    dataset_name: str
) -> None:
    """
    Save merged results to a JSON file.

    Args:
        results: Dictionary mapping sample IDs to model responses
        output_file: Path to the output JSON file
        dataset_name: Name of the dataset (e.g., 'ceval')
    """
    output = []
    sample_count = len(results)
    model_count = len(next(iter(results.values()))) if results else 0

    for sample_id, model_responses in results.items():
        prompt = next(iter(model_responses.values()))['prompt']
        for model, data in model_responses.items():
            assert data['prompt'] == prompt, f"Prompt mismatch for sample_id {sample_id}, model {model}"
        # 按CHINESE_RANKING排序模型名
        model_order = sorted(model_responses.keys(), key=lambda m: CODEING.get(m, 9999))
        resps = [model_responses[m]['response'] for m in model_order]
        models = list(model_order)

        sample_data = {
            'id': sample_id,
            'dataset': dataset_name,
            'prompt': prompt,
            'resps': resps,
            'models': models,
        }
        output.append(sample_data)

    output.sort(key=lambda x: x['id'])
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nMerged results saved to: {output_file}")
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {sample_count}")
    print(f"Models included: {model_count}")
    if results:
        print(f"Sample models: {', '.join(list(next(iter(results.values())).keys())[:3])}...")


def main():
    parser = argparse.ArgumentParser(
        description='Merge inference results from multiple models for a dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='outputs/mbpp',
        help='Directory containing model subdirectories with inference results'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='outputs/mbpp/merged_results.json',
        help='Output JSON file path'
    )

    args = parser.parse_args()

    try:
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
        print(f"Processing dataset: {dataset_name}")
        print(f"Looking for results in: {args.dataset_dir}")

        results = load_results(args.dataset_dir)
        if not results:
            print("No results found to merge.")
            return 1

        save_merged_results(results, args.output_file, dataset_name)
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
