import json
from pathlib import Path
import pandas as pd

def create_6x6_grid(resps):
    """
    将resps中的每个6位字符串分为6列，并组成6行×6列的表格。
    """
    if len(resps) < 6:
        raise ValueError("每个文档的resps必须至少包含6个6位字符串")
    
    grid = []
    for resp in resps[:6]:  # 只取前6个字符串
        #if len(resp) != 6 or not resp.isdigit():
            #raise ValueError(f"resps中的字符串必须是6位数字: {resp}")
        # 将6位字符串的每一位分成6列
        row = [int(char) for char in resp]
        grid.append(row)
    
    return grid

def export_to_excel(data, output_path):
    """
    将每个文档的数据导出到Excel文件，每个文档一个工作表。
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for doc in data:
            doc_id = doc.get('doc_id', 'unknown')
            resps = doc.get('resps', [])
            
            # 创建6x6表格
            grid = create_6x6_grid(resps)
            
            # 转换为DataFrame
            df = pd.DataFrame(grid, columns=[f"Column_{i+1}" for i in range(6)])
            
            # 确保工作表名称有效
            sheet_name = f"doc_{doc_id}"
            sheet_name = sheet_name[:31]  # Excel工作表名称限制
            
            # 写入到工作表
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def main():
    # 定义文件路径
    input_file = Path("ranks.json")
    output_excel = Path("ranks.xlsx")
    
    # 读取数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 导出到Excel
    export_to_excel(data, output_excel)
    print(f"已成功生成 {output_excel}")

if __name__ == "__main__":
    main()
