import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

def load_jsonl(file_path: str) -> List[dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def calculate_accuracy(data: List[dict]) -> Tuple[float, int, int]:
    """Calculate accuracy from data with 'correct' field."""
    correct = sum(1 for item in data if item.get('correct', False))
    total = len(data)
    return (correct / total * 100, correct, total)

def process_directory(directory: str) -> Dict[str, Tuple[float, int, int]]:
    """Process all JSONL files in a directory and calculate accuracies."""
    results = {}
    for root, _, files in os.walk(directory):
        jsonl_files = [f for f in files if f.endswith('.jsonl')]
        for file in jsonl_files:
            model_name = os.path.basename(os.path.dirname(os.path.join(root, file)))
            file_path = os.path.join(root, file)
            try:
                data = load_jsonl(file_path)
                accuracy, correct, total = calculate_accuracy(data)
                results[model_name] = (accuracy, correct, total)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return results

def print_results(results: Dict[str, Tuple[float, int, int]]) -> None:
    """Print results in a formatted table."""
    print("\n{:<40} {:<10} {:<10} {:<10}".format(
        "Model", "Accuracy", "Correct", "Total"))
    print("-" * 75)
    
    for model, (accuracy, correct, total) in sorted(results.items()):
        print("{:<40} {:<10.2f}% {:<10} {:<10}".format(
            model, accuracy, correct, total))
    
    # Calculate and print average accuracy
    if results:
        avg_accuracy = sum(acc for acc, _, _ in results.values()) / len(results)
        print("-" * 75)
        print("{:<40} {:<10.2f}%".format("Average Accuracy:", avg_accuracy))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate model accuracy from JSONL files.')
    parser.add_argument('directory', type=str, 
                       help='Directory containing model result JSONL files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        exit(1)
    
    results = process_directory(args.directory)
    print_results(results)
