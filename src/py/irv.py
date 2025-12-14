import pandas as pd
import numpy as np

def irv_elimination(rankings):
    """
    使用IRV排除法，输出候选人被淘汰的顺序。
    每轮选出排名最靠后的候选人淘汰。
    """
    candidates = list(range(1, rankings.shape[1] + 1)) # 初始候选人列表（1到n）
    elimination_order = [] # 用来保存每个候选人被淘汰的顺序

    while len(candidates) > 1:
    # 计算每个候选人在当前轮次的排名（1代表最优，6代表最差）
        ranks = {candidate: 0 for candidate in candidates}
        print(ranks)

        for ranking in rankings:
            # 累计每个候选人的排名位置，位置越大越不受欢迎（更靠后）
            for idx, candidate in enumerate(candidates):
                if candidate in ranking:
                    ranks[candidate] += ranking.tolist().index(candidate) + 1 # 1-based ranking

    # 找到排名最靠后的候选人（得分最高者为最差的）
        max_rank = max(ranks.values())
        candidates_to_eliminate = [candidate for candidate, rank in ranks.items() if rank == max_rank]

        # 将被淘汰的候选人加入淘汰顺序
        elimination_order.extend(candidates_to_eliminate)
        print(elimination_order)

        # 淘汰排名最靠后的候选人
        for candidate in candidates_to_eliminate:
            candidates.remove(candidate)

    # 重新调整排名：淘汰候选人的排名会消失
        new_rankings = []
        for ranking in rankings:
            new_ranking = [candidate for candidate in ranking if candidate in candidates]
            new_rankings.append(new_ranking)

        rankings = np.array(new_rankings)

    # 最终的胜者
    final_winner = candidates[0]

    # 添加胜者到淘汰顺序的最后
    elimination_order.append(final_winner)

    return elimination_order

file_path = r'/Users/msy/Desktop/ranks/mmlu_ranks.xlsx' # 替换为实际的文件路径
output_path = r'/Users/msy/Desktop/ranks/ranks_processed_mmlu_irv.xlsx' # 输出的Excel文件路径
# 处理多页 Excel 文件
excel_data = pd.ExcelFile(file_path)

# 创建一个新的 Excel 文件，保存每页的综合排名结果
with pd.ExcelWriter(output_path) as writer:
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
        optimal_rank = irv_elimination(rankings)
        
        # 将综合排名插入第 8 行
        if len(df) < 7:
            # 如果原始数据不足 7 行，填充空行
            while len(df) < 7:
                df.loc[len(df)] = [None] * df.shape[1]
        df.loc[7] = optimal_rank  # 第 8 行插入最优排名
        
        # 写入新的 Excel 文件
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"处理完成，结果已保存到 {output_file_path}")
