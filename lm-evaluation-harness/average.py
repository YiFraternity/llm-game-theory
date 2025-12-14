"""
计算并导出不同基准（gpqa、gsm8k_cot、mmlu）中模型排名的平均值与CHATBOT_ARENA_RANK（预设的模型排名）的皮尔逊相关性。

1. 计算每个模型的平均排名，并计算其与CHATBOT_ARENA_RANK（预设的模型排名）的皮尔逊相关系数。
2. 将所有基准测试的皮尔逊相关系数整理成一个DataFrame，并导出为Excel文件。导出时：
    - 每个基准测试占一列
    - 每行代表一个问题的皮尔逊相关系数
    - 最后一行是每个基准测试所有问题的平均皮尔逊相关系数
    - 平均值与问题结果之间用5行空值隔开，便于区分
3. 最终输出的Excel文件（average_pearson_correlations.xlsx）可以直观地比较不同基准测试中模型排名的相关性分布。

"""
import numpy as np
import pandas as pd

def create_6x6_grid(resps):
    grid = []
    for resp in resps[:6]:
        row = [int(char) for char in resp]
        grid.append(row)
    return grid


CHATBOT_ARENA_RANK = {
    'chatgpt-4o-latest-20241120': 1,
    'gpt-4o-2024-05-13': 2,
    'claude-3-5-sonnet-20241022': 3,
    'gpt-4o-2024-08-06': 4,
    'claude-3-5-haiku-20241022': 5,
    'claude-3-opus-20240229': 6,
}

def calc_average(file_path, output_path):
    """
    计算每个sheet的平均排名，返回每个sheet的pearson相关系数
    """
    excel_data = pd.read_excel(file_path, sheet_name=None)
    each_question_ranks_details = []
    total_ranks_avgs = []
    pearson_correlations = []
    for sheet_name, df in excel_data.items():

        if df.isnull().sum().sum() > 0:
            print(f"Skipping sheet '{sheet_name}' due to missing data.")
            continue # 如果有缺失值，跳过该sheet

        if df.shape[0] == 6 and df.shape[1] == 6:
            avg_rank = df.mean(axis=0)
            total_ranks_avgs.append(avg_rank)
            chatbot_arena_rank = [CHATBOT_ARENA_RANK[model] for model in avg_rank.index]
            pearson_correlation = np.corrcoef(chatbot_arena_rank, avg_rank)[0][1]
            each_question_ranks_details.append({
                'sheet_name': sheet_name,
                'avg_rank': avg_rank,
                'pearson_correlation': pearson_correlation,
            })
            pearson_correlations.append(pearson_correlation)
        else:
            print(f"Skipping sheet '{sheet_name}' due to invalid format or empty data.")
    each_question_ranks_df = pd.DataFrame(each_question_ranks_details)
    print(each_question_ranks_df.describe())
    total_avg_rank = np.mean(total_ranks_avgs, axis=0)
    print(total_avg_rank)
    print(chatbot_arena_rank)
    total_pearson_correlation = np.corrcoef(chatbot_arena_rank, total_avg_rank)[0][1]
    print(total_pearson_correlation)
    return pearson_correlations, total_pearson_correlation


if __name__ == "__main__":
    benchmarks = ['gpqa', 'gsm8k_cot', 'mmlu']
    # benchmarks = ['gsm8k_cot']
    pearson_list, total_pearson_list = [], []
    for benchmark in benchmarks:
        input_file = f'lm-evaluation-harness/output/{benchmark}_ranks.xlsx'
        pearson_correlations, total_pearson_correlation = calc_average(input_file, '')
        pearson_list.append(pearson_correlations)
        total_pearson_list.append(total_pearson_correlation)

    # 处理 pearson_list 里数组长度不一致
    max_len = max(len(p) for p in pearson_list)
    padded = [np.pad(np.array(p), (0, max_len - len(p)), constant_values=np.nan) for p in pearson_list]
    pearson_array = np.column_stack(padded)

    nan_rows = np.full((5, pearson_array.shape[1]), np.nan)
    total_row = np.array(total_pearson_list).reshape(1, -1)
    final_array = np.vstack([pearson_array, nan_rows, total_row])

    df = pd.DataFrame(final_array, columns=[f'{b}_pearson_correlation' for b in benchmarks])
    df.to_excel('lm-evaluation-harness/pearson_correl/average_pearson_correlations.xlsx', index=False)
