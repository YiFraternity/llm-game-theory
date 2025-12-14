
import itertools
import pandas as pd
import numpy as np


def compute_kemeny_young_ranking(file_path, output_path):
    # 读取Excel文件的所有Sheet
    excel_file = pd.ExcelFile(file_path)
    writer = pd.ExcelWriter(output_path, engine='openpyxl')  # 准备保存多Sheet的文件

    for sheet in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        if df.isnull().sum().sum() > 0:
            print(f"Skipping sheet '{sheet}' due to missing data.")
            continue  # 如果有缺失值，跳过该sheet

        # 提取列名中的候选人ID
        candidates = [int(col.split("_")[1]) if "_" in col else int(col) for col in df.columns.tolist()]

        # 构造偏好矩阵
        num_candidates = len(candidates)
        preferences = np.zeros((num_candidates, num_candidates))

        for row in df.itertuples(index=False):
            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    if row[i] < row[j]:
                        preferences[i][j] += 1
                    elif row[i] > row[j]:
                        preferences[j][i] += 1

        # 枚举所有可能的排名
        all_rankings = list(itertools.permutations(range(num_candidates)))
        min_distance = float('inf')
        best_ranking = None

        for ranking in all_rankings:
            distance = 0
            for i in range(num_candidates):
                for j in range(i + 1, num_candidates):
                    if ranking.index(i) < ranking.index(j):
                        distance += preferences[j][i]
                    else:
                        distance += preferences[i][j]
            if distance < min_distance:
                min_distance = distance
                best_ranking = ranking

        # 构造结果数据：每个候选人的排名位置（按原列顺序）
        rank_positions = [best_ranking.index(i) + 1 for i in range(num_candidates)]  # 从1开始排名
        result_data = [
            rank_positions,
            [min_distance] * num_candidates
        ]

        # 创建结果DataFrame
        result_df = pd.DataFrame(result_data, index=["Rank", "Minimum Distance"], columns=df.columns)

        # 将结果与原数据合并
        combined_df = pd.concat([df, result_df], axis=0)

        # 保存到对应的Sheet
        combined_df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()
    print(f"结果已保存至: {output_path}")

# 示例使用
file_path = r'py/ranks.xlsx'  # 替换为实际的文件路径
output_path = r'py/ranks_processed_kemeny.xlsx'  # 替换为保存结果的路径

compute_kemeny_young_ranking(file_path, output_path)