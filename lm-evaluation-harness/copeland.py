import pandas as pd
import numpy as np
from itertools import combinations

def validate_rankings(rankings, n_candidates):
    """验证排名数据有效性"""
    # 检查是否存在无效值
    if (rankings < 0).any() or (rankings >= n_candidates).any():
        invalid = np.unique(rankings[(rankings < 0) | (rankings >= n_candidates)])
        raise ValueError(f"包含无效候选人索引: {invalid}")

    # 检查每行是否为完整排列
    for i, vote in enumerate(rankings):
        unique = np.unique(vote)
        if len(unique) != n_candidates or not np.array_equal(np.sort(unique), np.arange(n_candidates)):
            raise ValueError(f"第{i+1}行不是完整排列")

def copeland_score(rankings):
    """安全的Copeland分数计算"""
    n_candidates = rankings.shape[1]
    validate_rankings(rankings, n_candidates)  # 新增数据验证
    
    scores = {c: 0 for c in range(n_candidates)}
    
    for a, b in combinations(range(n_candidates), 2):
        a_wins = b_wins = 0
        
        for vote in rankings:
            # 使用快速位置查询（替代np.where）
            a_pos = np.argmax(vote == a)
            b_pos = np.argmax(vote == b)
            
            # 验证位置有效性
            if vote[a_pos] != a or vote[b_pos] != b:
                raise RuntimeError("候选人位置解析错误")
            
            if a_pos < b_pos:
                a_wins += 1
            else:
                b_wins += 1
        
        # 更新分数
        if a_wins > b_wins:
            scores[a] += 1
            scores[b] -= 1
        elif a_wins < b_wins:
            scores[a] -= 1
            scores[b] += 1
    
    return scores

def copeland_ranking(df):
    """增强版Copeland排名生成"""
    # 数据清洗
    df = df.dropna(how='all').dropna(axis=1, how='all')
    if df.empty:
        return None
    
    # 获取候选人数
    n_cols = df.shape[1]
    
    # 数据验证与转换
    try:
        # 检查输入范围是否为1-based
        if not df.apply(lambda s: s.between(1, n_cols).all()).all():
            raise ValueError("输入数据包含超出范围的候选人编号")
            
        rankings = df.astype(int).subtract(1).to_numpy()
        validate_rankings(rankings, n_cols)
    except Exception as e:
        print(f"数据校验失败: {str(e)}")
        return None
    
    # 计算分数
    try:
        scores = copeland_score(rankings)
    except Exception as e:
        print(f"分数计算失败: {str(e)}")
        return None
    
    # 生成排序（分数相同按列顺序）
    sorted_indices = sorted(
        range(n_cols),
        key=lambda x: (-scores[x], x)
    )
    
    # 转回1-based编号
    return [i+1 for i in sorted_indices]

# 文件处理（保持不变）
input_path = '/Users/msy/Desktop/ranks/gpqa_ranks.xlsx'
output_path = '/Users/msy/Desktop/ranks/ranks_processed_gpqa_copeland_fixed.xlsx'

with pd.ExcelWriter(output_path) as writer:
    excel = pd.ExcelFile(input_path)
    
    for sheet_name in excel.sheet_names:
        df = excel.parse(sheet_name)
        result = copeland_ranking(df)
        
        if result:
            # 填充到原始列数
            padded = result + [None]*(df.shape[1]-len(result))
            
            # 插入到第8行
            while len(df) < 8:
                df.loc[len(df)] = [None]*df.shape[1]
            df.loc[7] = padded
        
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("处理完成，错误已修复")