
import os
import json
import copy
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from utils import CHATBOT_ARENA_RANK


def compute_average_ranks(grid):
    """
    计算每个模型（对角线元素）在其他行中的平均排名（四舍五入）
    """
    n = len(grid)
    diagonal_ranks = []

    for i in range(n):
        col_values = [grid[row][i] for row in range(n) if row != i]
        # avg_rank = round(sum(col_values) / len(col_values))
        avg_rank = sum(col_values) / len(col_values)
        diagonal_ranks.append(avg_rank)

    return diagonal_ranks

def create_6x6_grid(resps):
    """
    对于每一列，将对角线替换为均值（四舍五入），然后该列重新排序赋排名（1~6），
    排名时相对顺序保持不变。
    (若不四舍五入，尝试一下？，其实没必要必须是整数。)
    Args:
        resps: 6个模型的预测结果
    Returns:
        new_grid: 更新后的6x6网格
        raw_self_ranks: 每个模型自己对自己的排名
        updated_self_ranks: 每个模型别人对自己的排名
    """
    grid = []
    for resp in resps[:6]:
        row = [int(char) for char in resp]
        grid.append(row)

    raw_self_ranks, updated_self_ranks = [], []
    if len(grid) == 6 and all(len(row) == 6 for row in grid):
        diagonal_ranks = compute_average_ranks(grid)
        new_grid = copy.deepcopy(grid)
        for i in range(6):
            old_value = new_grid[i][i]
            new_rank = diagonal_ranks[i]
            raw_self_ranks.append(old_value)
            for j in range(6):
                if j == i:
                    continue
                val = new_grid[i][j]
                if new_rank < old_value:
                    if new_rank <= val and val < old_value:
                        new_grid[i][j] += 1
                elif new_rank > old_value:
                    if (old_value < val) and val <= new_rank:
                        new_grid[i][j] -= 1

            new_grid[i][i] = new_rank
            updated_self_ranks.append(new_rank)

        return new_grid, [raw_self_ranks, updated_self_ranks]
    else:
        return grid, None


def export_to_excel(data: List[Dict], raw_crowd_path: str, self_ranks_path: str):
    expected_models = data[0].get('models', [])
    assert all(item.get('models') == expected_models for item in data[1:]), \
           "Inconsistent models found in data"

    with pd.ExcelWriter(raw_crowd_path, engine='openpyxl') as writer:
        for doc in data:
            try:
                doc_id = doc.get('doc_id', 'unknown')
                resps = doc.get('resps', [])

                grid, _ = create_6x6_grid(resps)

                df = pd.DataFrame(grid, columns=expected_models)
                sheet_name = f"doc_{doc_id}"[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            except ValueError as e:
                print(f"跳过文档 {doc_id}，原因: {str(e)}")
                continue  # 跳过当前文档继续执行

    avg_self_ranks, avg_crowed_ranks = [], []
    problem_self_correl_list, problem_others_correl_list = [], []
    with pd.ExcelWriter(self_ranks_path, engine='openpyxl') as writer:
        for doc in data:
            try:
                doc_id = doc.get('doc_id', 'unknown')
                resps = doc.get('resps', [])

                _, self_ranks = create_6x6_grid(resps)
                if self_ranks:
                    avg_self_ranks.append(self_ranks[0])
                    avg_crowed_ranks.append(self_ranks[1])

                    chatbot_arena_rank = [CHATBOT_ARENA_RANK[model] for model in expected_models]
                    self_pearson_correlation = np.corrcoef(chatbot_arena_rank, self_ranks[0])[0][1]
                    problem_self_correl_list.append(self_pearson_correlation)
                    others_pearson_correlation = np.corrcoef(chatbot_arena_rank, self_ranks[1])[0][1]
                    problem_others_correl_list.append(others_pearson_correlation)

                df = pd.DataFrame(self_ranks, columns=expected_models)
                sheet_name = f"doc_{doc_id}"[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            except ValueError as e:
                print(f"跳过文档 {doc_id}，原因: {str(e)}")
                continue  # 跳过当前文档继续执行

        avg_self_ranks = np.mean(avg_self_ranks, axis=0)  # 按列平均
        avg_crowed_ranks = np.mean(avg_crowed_ranks, axis=0)  # 按列平均
        df = pd.DataFrame([avg_self_ranks, avg_crowed_ranks], columns=expected_models)
        # df.to_excel(writer, sheet_name="avg_ranks", index=False)
        self_avg_pearson_correlation = np.corrcoef(chatbot_arena_rank, avg_self_ranks)[0][1]
        others_avg_pearson_correlation = np.corrcoef(chatbot_arena_rank, avg_crowed_ranks)[0][1]

        return df, [
            problem_self_correl_list,
            problem_others_correl_list,
            self_avg_pearson_correlation,
            others_avg_pearson_correlation,
        ]


if __name__ == "__main__":
    ROOT_PATH = 'lm-evaluation-harness/output/'
    benchmarks = os.listdir(ROOT_PATH)
    benchmarks = [benchmark for benchmark in benchmarks if os.path.isdir(os.path.join(ROOT_PATH, benchmark))]
    # benchmarks = ['gsm8k_cot']
    # benchmarks = ['gpqa', 'mmlu']
    print(benchmarks)
    dfs = []
    benchmark_self_correl_list = []
    benchmark_others_correl_list = []
    benchmark_self_avg_pearson_correl_list = []
    benchmark_others_avg_pearson_correl_list = []
    for benchmark in benchmarks:
        input_file = Path(ROOT_PATH, benchmark, 'rank', 'input', 'ranks.json')
        crowd_ranks_path = Path(ROOT_PATH, f'{benchmark}_crowd_ranks.xlsx')
        self_ranks_path = Path(ROOT_PATH, f'{benchmark}_self_ranks.xlsx')

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        df, correl_list = export_to_excel(data, crowd_ranks_path, self_ranks_path)
        dfs.append(df)
        benchmark_self_correl_list.append(correl_list[0])
        benchmark_others_correl_list.append(correl_list[1])
        benchmark_self_avg_pearson_correl_list.append(correl_list[2])
        benchmark_others_avg_pearson_correl_list.append(correl_list[3])

    dfs = pd.concat(dfs, axis=0)
    print(dfs.columns)
    reindex_columns = [
        'chatgpt-4o-latest-20241120',
        'gpt-4o-2024-05-13',
        'claude-3-5-sonnet-20241022',
        'gpt-4o-2024-08-06',
        'claude-3-5-haiku-20241022',
        'claude-3-opus-20240229']
    dfs = dfs.reindex(columns=reindex_columns)
    new_columns = ['4o-latest', '40-0513', 'sonnet-1022', '40-0806', 'haiku-1022', 'opus-0229']
    dfs.columns = new_columns
    dfs.to_excel('lm-evaluation-harness/output/all_self_crowd_ranks.xlsx', index=False)
    print("处理完成，已跳过异常数据，结果已保存到", 'lm-evaluation-harness/output/all_self_crowd_ranks.xlsx')

    # 处理 pearson_list 里数组长度不一致
    # 箱形图展示，每个问题对应的皮尔逊相关系数，以及平均值的皮尔逊相关系数
    max_len = max(len(p) for p in benchmark_self_correl_list)
    padded = [np.pad(np.array(p), (0, max_len - len(p)), constant_values=np.nan) for p in benchmark_self_correl_list]
    pearson_array = np.column_stack(padded)

    nan_rows = np.full((5, pearson_array.shape[1]), np.nan)
    total_row = np.array(benchmark_self_avg_pearson_correl_list).reshape(1, -1)
    final_array = np.vstack([pearson_array, nan_rows, total_row])

    df = pd.DataFrame(final_array, columns=[f'{b}_pearson_correlation' for b in benchmarks])
    df.to_excel('lm-evaluation-harness/pearson_correl/self_pearson_correlations.xlsx', index=False)

    max_len = max(len(p) for p in benchmark_others_correl_list)
    padded = [np.pad(np.array(p), (0, max_len - len(p)), constant_values=np.nan) for p in benchmark_others_correl_list]
    pearson_array = np.column_stack(padded)

    nan_rows = np.full((5, pearson_array.shape[1]), np.nan)
    total_row = np.array(benchmark_others_avg_pearson_correl_list).reshape(1, -1)
    final_array = np.vstack([pearson_array, nan_rows, total_row])

    df = pd.DataFrame(final_array, columns=[f'{b}_pearson_correlation' for b in benchmarks])
    df.to_excel('lm-evaluation-harness/pearson_correl/others_pearson_correlations.xlsx', index=False)
