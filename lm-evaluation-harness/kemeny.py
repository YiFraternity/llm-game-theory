import pandas as pd
import itertools
import numpy as np

def compute_kemeny_young_ranking(file_path, output_path):
    # 读取Excel文件的所有Sheet
    excel_file = pd.ExcelFile(file_path)
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    for sheet in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        
        # 跳过包含缺失值的sheet
        if df.isnull().sum().sum() > 0:
            print(f"Skipping sheet '{sheet}' due to missing data.")
            continue

        # 直接使用原始列名作为候选人标识
        candidates = df.columns.tolist()  # 修改点：保留原始列名

        # 构造偏好矩阵（使用列索引代替候选人标识）
        num_candidates = len(candidates)
        preferences = np.zeros((num_candidates, num_candidates))

        for row in df.itertuples(index=False):
            for i in range(num_candidates):
                for j in range(i+1, num_candidates):
                    if row[i] < row[j]:
                        preferences[i][j] += 1
                    elif row[i] > row[j]:
                        preferences[j][i] += 1

        # 枚举所有可能的排名（基于列索引）
        all_rankings = list(itertools.permutations(range(num_candidates)))
        min_distance = float('inf')
        best_ranking = None

        for ranking in all_rankings:
            distance = 0
            for i in range(num_candidates):
                for j in range(i+1, num_candidates):
                    if ranking.index(i) < ranking.index(j):
                        distance += preferences[j][i]
                    else:
                        distance += preferences[i][j]
            if distance < min_distance:
                min_distance = distance
                best_ranking = ranking

        # 构建结果数据（保留原始列名）
        rank_order = [candidates[i] for i in best_ranking]  # 按最优排序的列名
        result_data = [
            [rank_order.index(col) + 1 for col in candidates],  # 计算每个列的排名位置
            [min_distance] * num_candidates
        ]

        # 创建结果DataFrame
        result_df = pd.DataFrame(
            result_data,
            index=["Rank Position", "Kemeny Distance"],
            columns=candidates
        )

        # 合并原始数据与结果
        combined_df = pd.concat([df, result_df], axis=0)
        
        # 保存到sheet
        combined_df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()
    print(f"处理结果已保存至: {output_path}")

# 使用示例
file_path = r'/Users/msy/Desktop/ranks/gpqa_ranks.xlsx'
output_path = r'/Users/msy/Desktop/ranks/ranks_processed_gpqa_kemeny.xlsx'

compute_kemeny_young_ranking(file_path, output_path)