import json
import os

def replace_ranks(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each entry in the data
    for entry in data:
        if 'resps' in entry:
            new_resps = []
            for rank_str in entry['resps']:
                # Replace any digit > '3' with '9'
                new_rank = ''.join(['9' if c > '3' and c.isdigit() else c for c in rank_str])
                new_resps.append(new_rank)
            entry['resps'] = new_resps

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the processed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'outputs/gsm8k/input/ranks.json')
    output_file = os.path.join(base_dir, 'outputs/gsm8k/input/ranks_top3.json')

    print(f"Processing {input_file}...")
    replace_ranks(input_file, output_file)
    print(f"Processing complete. Results saved to {output_file}")
