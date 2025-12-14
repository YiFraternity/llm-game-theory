import pandas as pd
from itertools import permutations
from scipy.stats import spearmanr

def compute_total_spearman(combined_rank, rankings):
    """
    计算综合排名与多个排名的 Kendall Tau 系数之和
    """
    total_spearman = 0
    for rank in rankings:  # 按行比较
        tau, _ = spearmanr(combined_rank, rank)
        total_spearman += tau  # 取绝对值，衡量一致性
    return total_spearman

def find_optimal_combined_rank(rankings):
    """
    寻找综合排名，使得 Kendall Tau 系数之和最小
    """
    n = rankings.shape[1]  # 排名元素数量
    all_permutations = list(permutations(range(1, n+1)))  # 所有可能的排名
    min_kendall = 0
    best_rank = None

    for perm in all_permutations:
        total_kendall = compute_total_spearman(perm, rankings)
        if total_kendall > min_kendall:
            min_kendall = total_kendall
            best_rank = perm

    return best_rank, min_kendall

# 输入和输出文件路径
input_file_path = r'/Users/msy/Desktop/ranks/gpqa_ranks.xlsx'  # 替换为实际文件路径
output_file_path = r'/Users/msy/Desktop/ranks/ranks_processed_gpqa_spearman.xlsx'  # 输出文件路径

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
        optimal_rank, min_kendall_sum = find_optimal_combined_rank(rankings)
        
        # 将综合排名插入第 8 行
        if len(df) < 7:
            # 如果原始数据不足 7 行，填充空行
            while len(df) < 7:
                df.loc[len(df)] = [None] * df.shape[1]
        df.loc[7] = optimal_rank  # 第 8 行插入最优排名
        
        # 写入新的 Excel 文件
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"页 {sheet_name} 处理完成，最小 spearman 和为 {min_kendall_sum}")

print(f"处理完成，结果已保存到 {output_file_path}")