import pandas as pd
import numpy as np
from itertools import permutations

def count_dodgson_swaps(rankings, candidate_index):
    """
    计算候选人成为第一名所需要的最少交换次数
    """
    # 计算候选人成为第一名时需要的交换次数
    swaps = 0
    for ranking in rankings:
        current_pos = np.where(ranking == candidate_index)[0][0]
        swaps += current_pos # 交换次数就是候选人的当前位置
    return swaps

def find_optimal_combined_rank(rankings):
    """
    使用Dodgson方法计算最优综合排名
    """
    n = rankings.shape[1] # 排名元素数量
    candidate_indices = range(1,n+1) # 所有候选人索引

    # 为每个候选人计算所需的交换次数
    swap_counts = []
    for candidate in candidate_indices:
        swaps_for_candidate = count_dodgson_swaps(rankings, candidate)
        swap_counts.append((candidate, swaps_for_candidate))

    # 按照交换次数排序，从少到多
    swap_counts.sort(key=lambda x: x[1])

    # 得到排序后的候选人
    optimal_rank = [candidate for candidate, _ in swap_counts]
    swap_times = [swaps for _, swaps in swap_counts]

    return optimal_rank, swap_times

# 输入和输出文件路径
input_file_path = r'/Users/msy/Desktop/ranks/mmlu_ranks.xlsx' # 替换为实际文件路径
output_file_path = r'/Users/msy/Desktop/ranks/ranks_processed_borda.xlsx'  # 输出文件路径

# 处理多页 Excel 文件
excel_data = pd.ExcelFile(input_file_path)

# 创建一个新的 Excel 文件，保存每页的综合排名结果
with pd.ExcelWriter(output_file_path) as writer:
    for sheet_name in excel_data.sheet_names:
    # 读取当前页的数据
        df = excel_data.parse(sheet_name)

        # 检查数据是否存在缺失值
        if df.isnull().values.any():
            print(f"跳过页 {sheet_name}，因数据存在缺失值。")
            # 直接保存原始数据
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            continue

        # 转换为 NumPy 数组进行计算
        rankings = df.to_numpy()

        # 求解最优综合排名
        optimal_rank, cost = find_optimal_combined_rank(rankings)

        # 将综合排名插入第 7 行
        if len(df) < 6: # 如果原始数据不足 6 行，填充空行
            while len(df) < 6:
                df.loc[len(df)] = [None] * df.shape[1]
        df.loc[6] = optimal_rank # 第 7 行插入最优排名
        df.loc[7] = cost

        # 写入新的 Excel 文件
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"页 {sheet_name} 处理完成")

print(f"处理完成，结果已保存到 {output_file_path}")