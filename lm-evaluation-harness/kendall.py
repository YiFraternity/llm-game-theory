from itertools import permutations
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from utils import CHATBOT_ARENA_RANK


def compute_total_kendall(combined_rank, rankings):
    """
    计算综合排名与多个排名的 Kendall Tau 系数之和
    """
    total_kendall = 0
    for rank in rankings:  # 按行比较
        tau, _ = kendalltau(combined_rank, rank)
        total_kendall += tau  # 取绝对值，衡量一致性
    return total_kendall


def find_optimal_combined_rank(rankings):
    n = rankings.shape[1]  # 排名元素数量
    all_permutations = list(permutations(range(1, n+1)))  # 所有可能的排名
    min_kendall = 0
    best_rank = None

    for perm in all_permutations:
        total_kendall = compute_total_kendall(perm, rankings)
        if total_kendall > min_kendall:
            min_kendall = total_kendall
            best_rank = perm

    return best_rank, min_kendall


def process_excel_sheets(input_file_path, output_file_path, final_avg_file_path):
    """
    处理多页 Excel 文件，针对每个sheet计算最优综合排名，将结果写入新Excel，并返回每个sheet的最终排名。
    返回: dict, key为sheet名，value为(最优排名, min_kendall_sum)或None（如果跳过）
    """
    excel_data = pd.ExcelFile(input_file_path)
    sheet_results = {}
    final_avg_list = []
    expected_models = []
    each_problem_kendall_correlation = []
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name in excel_data.sheet_names:
            # 读取当前页的数据
            df = excel_data.parse(sheet_name)
            expected_models = df.columns.tolist()
            # 检查数据是否存在缺失值
            if df.isnull().values.any():
                print(f"跳过页 {sheet_name}，因数据存在缺失值。")
                # 直接保存原始数据
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                sheet_results[sheet_name] = None
                continue

            rankings = df.to_numpy()

            # 求解最优综合排名
            chatbot_arena_rank = [CHATBOT_ARENA_RANK[model] for model in df.columns]
            optimal_rank, min_kendall_sum = find_optimal_combined_rank(rankings)
            correlation = np.corrcoef(chatbot_arena_rank, optimal_rank)[0][1]
            each_problem_kendall_correlation.append({
                'sheet_name': sheet_name,
                'optimal_rank': optimal_rank,
                'correlation': correlation,
            })
            # 将综合排名插入第 8 行
            if len(df) < 7:
                # 如果原始数据不足 7 行，填充空行
                while len(df) < 7:
                    df.loc[len(df)] = [None] * df.shape[1]
            df.loc[7] = optimal_rank  # 第 8 行插入最优排名
            final_avg_list.append(optimal_rank)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"页 {sheet_name} 处理完成，最小 Kendall Tau 和为 {min_kendall_sum}")
            sheet_results[sheet_name] = (optimal_rank, min_kendall_sum)
    print(f"处理完成，结果已保存到 {output_file_path}")

    # 按列平均
    final_avg = np.mean(final_avg_list, axis=0)
    df = pd.DataFrame([final_avg], columns=expected_models)
    df.to_excel(final_avg_file_path, index=False)
    problem_kendall_correlation_df = pd.DataFrame(each_problem_kendall_correlation)
    print(problem_kendall_correlation_df.describe())
    return sheet_results

if __name__ == "__main__":
    input_file_path = [
        # 'lm-evaluation-harness/output/gpqa_ranks.xlsx',
        # 'lm-evaluation-harness/output/gpqa_crowd_ranks.xlsx',
        # 'lm-evaluation-harness/output/gsm8k_cot_ranks.xlsx',
        # 'lm-evaluation-harness/output/gsm8k_cot_crowd_ranks.xlsx',
        'lm-evaluation-harness/output/mmlu_ranks.xlsx',
        # 'lm-evaluation-harness/output/mmlu_crowd_ranks.xlsx',
    ]
    output_file_path = [
        'rerank/output/kendall/mmlu_ranks_kendall.xlsx',
        # 'rerank/output/kendall/gpqa_crowd_ranks_kendall.xlsx',
        # 'rerank/output/kendall/gsm8k_cot_ranks_kendall.xlsx',
        # 'rerank/output/kendall/gsm8k_cot_crowd_ranks_kendall.xlsx',
        # 'rerank/output/kendall/mmlu_ranks_kendall.xlsx',
        # 'rerank/output/kendall/mmlu_crowd_ranks_kendall.xlsx',
    ]
    final_avg_file_path = [
        'rerank/output/kendall/mmlu_ranks_kendall_avg.xlsx',
        # 'rerank/output/kendall/gpqa_crowd_ranks_kendall_avg.xlsx',
        # 'rerank/output/kendall/gsm8k_cot_ranks_kendall_avg.xlsx',
        # 'rerank/output/kendall/gsm8k_cot_crowd_ranks_kendall_avg.xlsx',
        # 'rerank/output/kendall/mmlu_ranks_kendall_avg.xlsx',
        # 'rerank/output/kendall/mmlu_crowd_ranks_kendall_avg.xlsx',
    ]
    for input_path, output_path, final_avg_path in zip(input_file_path, output_file_path, final_avg_file_path):
        process_excel_sheets(input_path, output_path, final_avg_path)
