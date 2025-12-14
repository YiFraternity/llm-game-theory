import pandas as pd
import numpy as np

def irv_ranking_with_order(rankings):
    """生成IRV淘汰顺序的排名映射"""
    # 初始化参数
    n_candidates = rankings.shape[1]
    candidates = list(range(1, n_candidates+1))
    active = candidates.copy()
    elimination_order = []  # 淘汰顺序（不含胜者）

    # 运行IRV淘汰过程
    current_rankings = rankings.copy()
    while len(active) > 1:
        # 统计第一选择票数
        vote_counts = {c:0 for c in active}
        for vote in current_rankings:
            for candidate in vote:
                if candidate in active:
                    vote_counts[candidate] += 1
                    break

        # 找出最低票候选人
        min_votes = min(vote_counts.values())
        to_eliminate = [c for c in vote_counts if vote_counts[c] == min_votes]

        # 处理平票
        if len(active) - len(to_eliminate) >= 1:
            elimination_order.extend(to_eliminate)
            active = [c for c in active if c not in to_eliminate]
        else:
            elimination_order.extend(active[:-1])
            active = [active[-1]]

        # 更新投票数据
        current_rankings = np.array([
            [c for c in vote if c in active]
            for vote in current_rankings
        ])

    # 最终排名顺序：胜者 + 淘汰逆序
    final_order = active + elimination_order[::-1]

    # 生成排名字典 {候选人: 排名}
    return {candidate: rank
            for rank, candidate in enumerate(final_order, 1)}

def process_sheet_corrected(df):
    """处理工作表并生成正确排名"""
    # 数据清洗
    df = df.dropna(how='all').dropna(axis=1, how='all')
    if df.empty:
        return None

    # 转换为数字矩阵
    try:
        rankings = df.astype(int).to_numpy()
    except:
        return None

    # 获取IRV排名映射
    rank_map = irv_ranking_with_order(rankings)

    # 按原始列顺序生成排名（假设列顺序为1到N）
    original_columns = list(range(1, df.shape[1]+1))
    return [rank_map.get(col, None) for col in original_columns]

# 文件处理主程序
input_path = 'lm-evaluation-harness/output/gpqa_ranks.xlsx'
output_path = 'lm-evaluation-harness/output/ranks_processed_gpqa_irv_ranks.xlsx'

with pd.ExcelWriter(output_path) as writer:
    excel = pd.ExcelFile(input_path)

    for sheet_name in excel.sheet_names:
        # 读取数据（保留原始列顺序）
        df = excel.parse(sheet_name)

        # 处理数据
        result = process_sheet_corrected(df)

        # 插入结果到第8行
        if result:
            # 填充结果到原始列数
            padded_result = result + [None] * (df.shape[1] - len(result))

            # 确保有至少8行
            while len(df) < 8:
                df.loc[len(df)] = [None] * df.shape[1]

            # 插入到第8行（索引7）
            df.loc[7] = padded_result

        # 保存结果
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("处理完成，正确排名已保存至:", output_path)
