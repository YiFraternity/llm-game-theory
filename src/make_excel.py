
import os
import json
from typing import List, Dict
from pathlib import Path
import pandas as pd

def create_6x6_grid(resps):
    grid = []
    for resp in resps[:6]:
        row = [int(char) for char in resp]
        grid.append(row)
    return grid

def export_to_excel(data: List[Dict], output_path: str):
    expected_models = data[0].get('models', [])
    assert all(item.get('models') == expected_models for item in data[1:]), \
           "Inconsistent models found in data"

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for doc in data:
            try:
                doc_id = doc.get('doc_id', 'unknown')
                resps = doc.get('resps', [])

                grid = create_6x6_grid(resps)

                # 添加维度校验
                if (len(grid) != 6) or any(len(row) != 6 for row in grid):
                    raise ValueError(f"Invalid grid dimensions for doc {doc_id}")

                df = pd.DataFrame(grid, columns=expected_models)
                sheet_name = f"doc_{doc_id}"[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            except ValueError as e:
                print(f"跳过文档 {doc_id}，原因: {str(e)}")
                continue  # 跳过当前文档继续执行

def main(input_file, output_excel):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    export_to_excel(data, output_excel)
    print("处理完成，已跳过异常数据，结果已保存到", output_excel)

if __name__ == "__main__":
    ROOT_PATH = 'outputs/'
    # benchmarks = os.listdir(ROOT_PATH)
    benchmarks = ['gsm8k']
    benchmarks = [benchmark for benchmark in benchmarks if os.path.isdir(os.path.join(ROOT_PATH, benchmark))]
    for benchmark in benchmarks:
        input_file = Path(ROOT_PATH, benchmark, 'input', 'ranks.json')
        output_excel = Path(ROOT_PATH, f'{benchmark}_ranks.xlsx')
        main(input_file, output_excel)
