import pandas as pd
import numpy as np

def enhanced_dodgson_processor(df):
    """增强版Dodgson处理器（输出原始列号）"""
    # 数据清洗
    df_clean = df.dropna(how='all').dropna(axis=1, how='all')
    if df_clean.empty:
        raise ValueError("有效数据为空")

    # 获取原始列结构
    original_columns = df_clean.columns.tolist()
    n_cols = len(original_columns)

    # 验证数据有效性
    valid_rows = df_clean.apply(lambda row: row.nunique() == n_cols, axis=1)
    df_valid = df_clean[valid_rows]
    
    # 生成列号映射表（1-based）
    col_mapping = {col: idx+1 for idx, col in enumerate(original_columns)}

    # 转换为数值矩阵
    try:
        rankings = df_valid.replace(col_mapping).to_numpy(int)
    except Exception as e:
        raise ValueError(f"数据转换失败: {str(e)}")

    # 计算每个列的交换次数
    swap_counts = []
    for col in original_columns:
        code = col_mapping[col]
        total_swaps = sum(np.where(rank == code)[0][0] for rank in rankings)
        swap_counts.append(total_swaps)

    # 生成排序结果（保持原始列顺序）
    sorted_indices = np.argsort(swap_counts)
    ranked_order = [i+1 for i in sorted_indices]  # 转换为1-based编号
    sorted_swaps = [swap_counts[i] for i in sorted_indices]

    return ranked_order, sorted_swaps

# 文件处理
input_path = r'/Users/msy/Desktop/ranks/gsm8k_ranks.xlsx' # 替换为实际的文件路径
output_path = r'/Users/msy/Desktop/ranks/ranks_processed_gsm8k_dodgson.xlsx' # 输出的Excel文件路径

with pd.ExcelWriter(output_path) as writer:
    excel = pd.ExcelFile(input_path)
    
    for sheet in excel.sheet_names:
        try:
            # 读取数据（保留原始列结构）
            raw_df = excel.parse(sheet, header=0)
            
            # 执行处理
            order, swaps = enhanced_dodgson_processor(raw_df)
            
            # 构建结果行（确保列对齐）
            result_row = order + [None]*(len(raw_df.columns)-len(order))
            cost_row = swaps + [None]*(len(raw_df.columns)-len(swaps))
            
            # 插入到第8行
            output_df = raw_df.copy()
            output_df.loc[7] = result_row  # 第8行（0-based索引7）
            output_df.loc[8] = cost_row
            
            output_df.to_excel(writer, sheet_name=sheet, index=False)
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            raw_df.to_excel(writer, sheet_name=sheet, index=False)

print("处理完成，结果已保存")