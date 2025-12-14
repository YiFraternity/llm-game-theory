import pandas as pd
import numpy as np

def condorcet_method(data):
    """
    使用孔多塞法对候选人进行排名，输入为一个候选人之间的排名矩阵（6x6），返回排名。
    每行表示一个选民的投票，列表示候选人的顺序。
    """
    num_candidates = len(data.columns) # 假设每一列代表一个候选人，列数为候选人数（这里是6）

    # 存储每对候选人之间的胜负结果，1表示A胜B，-1表示B胜A，0表示平局
    victories = np.zeros((num_candidates, num_candidates), dtype=int)

    # 对每一对候选人进行比较
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            i_wins = 0
            j_wins = 0
            for _, row in data.iterrows():
            # 对每个选民的投票，判断候选人i和候选人j的排名
                if row.iloc[i] < row.iloc[j]: # i排在j前面，i胜
                    i_wins += 1
                elif row.iloc[j] < row.iloc[i]: # j排在i前面，j胜
                    j_wins += 1
            # 如果相同（平局），什么都不做
                if i_wins > j_wins:
                    victories[i][j] = 1
                    victories[j][i] = -1
                elif j_wins > i_wins:
                    victories[j][i] = 1
                    victories[i][j] = -1
                else:
                    victories[i][j] = 0
                    victories[j][i] = 0

    # 计算每个候选人的胜利数量（即赢得的配对数量）
    scores = np.sum(victories == 1, axis=1)

    # 根据得分对候选人排序（得分高的排前）
    ranking = np.argsort(-scores) # 降序排序
    return scores, ranking

def rank_excel_sheets(file_path, output_path):
    """
    读取Excel文件的每一页数据，并使用孔多塞法对每一页的候选人进行排名。
    将源数据、得分和排名结果保存到同一个Excel文件中的不同sheet。
    """
    # 读取Excel文件
    excel_data = pd.read_excel(file_path, sheet_name=None)

    # 创建ExcelWriter对象，用于将结果写入同一个Excel文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 遍历每一页(sheet)
        for sheet_name, df in excel_data.items():

            if df.isnull().sum().sum() > 0:
                print(f"Skipping sheet '{sheet_name}' due to missing data.")
                continue # 如果有缺失值，跳过该sheet

        # 确保sheet是6x6的矩阵
            if df.shape[0] == 6 and df.shape[1] == 6:
            # 由于Excel中的每页数据为6x6矩阵，行是选民，列是候选人
                data = df

                # 使用孔多塞法进行排名，返回得分和排名
                scores, ranking = condorcet_method(data)
                scores_row = pd.DataFrame(scores, index=data.columns)
                ranking_column = pd.DataFrame(ranking + 1, index=data.columns)

                # 将源数据、得分和排名合并到一个DataFrame
                result_df = pd.concat([data, scores_row.T, ranking_column.T], axis=0)

                result_df.reset_index(drop=True, inplace=True)

                # 保存该sheet的所有结果到Excel文件
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)

            else:
                print(f"Skipping sheet '{sheet_name}' due to invalid format or empty data.")

# 使用示例
file_path = r'/Users/msy/Desktop/ranks/mmlu_ranks.xlsx' # 替换为实际的文件路径
output_path = r'/Users/msy/Desktop/ranks/ranks_processed_borda.xlsx' # 输出的Excel文件路径
rank_excel_sheets(file_path, output_path)