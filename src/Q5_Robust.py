"""
计算并导出不同基准（gpqa、gsm8k_cot、mmlu）中模型排名的平均值与CHATBOT_ARENA_RANK（预设的模型排名）的皮尔逊相关性。

1. 计算每个模型的平均排名，并计算其与CHATBOT_ARENA_RANK（预设的模型排名）的皮尔逊相关系数。
2. 将所有基准测试的皮尔逊相关系数整理成一个DataFrame，并导出为Excel文件。导出时：
    - 每个基准测试占一列
    - 每行代表一个问题的皮尔逊相关系数
    - 最后一行是每个基准测试所有问题的平均皮尔逊相关系数
    - 平均值与问题结果之间用5行空值隔开，便于区分
3. 最终输出的Excel文件（average_pearson_correlations.xlsx）可以直观地比较不同基准测试中模型排名的相关性分布。

"""
import os
from typing import List
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import kendalltau, pearsonr
from utils import (
    load_json_data,
    CHINESE_RANKING,
    INSTRUCTION_RANKING,
    CREATING_WRITING,
    CODEING,
    MATHEMATIC,
    OVERALL_RANKING,
    MMLU_MODEL_ID_NAME,
    MBPP_MODEL_ID_NAME,
    CEVAL_MODEL_ID_NAME,
    GSM8K_MODEL_ID_NAME,
    GPQA_MODEL_ID_NAME,
    WRITING_BENCH_MODEL_ID_NAME,
    IFEVAL_MODEL_ID_NAME,
    get_rank_matrix,
    Rerank,
)

def rerank_matrix_using_df(method: str, rank_df: pd.DataFrame) -> List:
    """
    与main方法类似，但是输入是DataFrame，而不是excel文件
    计算dataframe的重排后的结果
    Args:
        method: rerank方法, kendall, borda, average, spearman, kemeny_young, irv
        rank_df: DataFrame
    Returns:
        rerank_result: List[float] 重排后的结果
    """
    if rank_df.shape[0] == 6 and rank_df.shape[1] == 6:
        rankings = rank_df.values.tolist()
        rerank = Rerank(rankings)
        rerank_result = rerank.rerank_method(method)
        return rerank_result
    else:
        print(f"Skipping sheet due to invalid format or empty data.")
        return None

def main(method, file_path, chatbot_arena):
    """
    计算每个sheet的平均排名，返回每个sheet的pearson相关系数

    Args:
        method: rerank方法, kendall, borda, average, spearman, kemeny_young, irv
        file_path: excel文件路径
    Returns:
        pearson_correlations: List[float] 每个sheet重新排序后的与CHATBOT_ARENA_RANK（预设的模型排名）的pearson相关系数
        total_pearson_correlation: float 所有sheet重新排序后的与CHATBOT_ARENA_RANK（预设的模型排名）的平均pearson相关系数
    """
    excel_data = pd.read_excel(file_path, sheet_name=None)
    each_question_ranks_details = []
    total_ranks_avgs = []
    pearson_correlations = []
    for sheet_name, df in excel_data.items():

        if df.isnull().sum().sum() > 0:
            print(f"Skipping sheet '{sheet_name}' due to missing data.")
            continue # 如果有缺失值，跳过该sheet

        if df.shape[0] == 6 and df.shape[1] == 6:
            rankings = df.values.tolist()
            model_names = list(df.columns)
            rerank = Rerank(rankings)
            rerank_ranking = rerank.rerank_method(method)
            total_ranks_avgs.append(rerank_ranking)

            chatbot_arena_rank = [chatbot_arena[model] for model in model_names]
            pearson_correlation = np.corrcoef(chatbot_arena_rank, rerank_ranking)[0][1]

            each_question_ranks_details.append({
                'method': method,
                'sheet_name': sheet_name,
                'rerank_ranking': rerank_ranking,
                'pearson_correlation': pearson_correlation,
            })
            pearson_correlations.append(pearson_correlation)
        else:
            print(f"Skipping sheet '{sheet_name}' due to invalid format or empty data.")
    each_question_ranks_df = pd.DataFrame(each_question_ranks_details)
    # 保留 4 位小数输出描述性统计
    try:
        print(each_question_ranks_df.describe().round(4).to_markdown())
    except Exception:
        # 回退：如果不能生成 markdown（例如没有数值列），仍然打印原始描述但四舍五入
        print(each_question_ranks_df.describe().round(4))
    total_rerank_ranking = np.mean(total_ranks_avgs, axis=0)
    # 四位小数输出总体重排排名（保留为列表便于阅读）
    try:
        print(np.round(total_rerank_ranking, 4).tolist())
    except Exception:
        print(total_rerank_ranking)
    print(chatbot_arena_rank)
    total_pearson_correlation = np.corrcoef(chatbot_arena_rank, total_rerank_ranking)[0][1]
    print(f"{total_pearson_correlation:.4f}")
    return pearson_correlations, total_pearson_correlation


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--input-template', default='outputs/{bench}/input/ranks_perturbed_q{q}.json',
                   help="Input file template with {bench} placeholder, e.g. outputs/{bench}/input/ranks_perturbed_q0.3.json")
    p.add_argument('--bench', default='gsm8k', help='optional comma-separated list of benchmarks to run (e.g. gsm8k)')
    args = p.parse_args()

    benchmarks = ['ceval', 'ifeval', 'mbpp', 'writingbench', 'gsm8k', 'gpqa', 'mmlu']
    if args.bench:
        benchmarks = [b.strip() for b in args.bench.split(',') if b.strip()]
    bench_capability = {
        'mmlu': OVERALL_RANKING,
        'mbpp': CREATING_WRITING,
        'ifeval': INSTRUCTION_RANKING,
        'writingbench': CODEING,
        'ceval': CHINESE_RANKING,
        'gpqa': OVERALL_RANKING,
        'gsm8k': MATHEMATIC,
    }
    model_id_name = {
        'mmlu': MMLU_MODEL_ID_NAME,
        'mbpp': MBPP_MODEL_ID_NAME,
        'ifeval': IFEVAL_MODEL_ID_NAME,
        'writingbench': WRITING_BENCH_MODEL_ID_NAME,
        'ceval': CEVAL_MODEL_ID_NAME,
        'gpqa': GPQA_MODEL_ID_NAME,
        'gsm8k': GSM8K_MODEL_ID_NAME,
    }
    methods = ['kemeny_young']

    model_macro_kendall, model_macro_pearson = {method: {} for method in methods}, {method: {} for method in methods}
    for bench in benchmarks:
        has_q = '{q}' in args.input_template
        qs = [round(i * 0.1, 1) for i in range(11)] if has_q else [None]

        # will collect per-question kendall per q
        perq_kendall_by_q = {}
        perq_doc_ids = None

        for q_val in qs:
            if q_val is None:
                input_file = args.input_template.format(bench=bench)
            else:
                # format q as 0.3 etc. The template is expected to include {q}
                input_file = args.input_template.format(bench=bench, q=f"{q_val:.1f}")

            # load perturbed (or original) ranks json
            data = load_json_data(input_file)
            rank_matrixs = get_rank_matrix(data, model_id_name[bench], bench_capability[bench])

            method = 'kemeny_young'
            per_q_kendall = []
            doc_ids = []

            # iterate entries and corresponding rank matrices
            for entry, rank_df in tqdm(list(zip(data, rank_matrixs)), desc=f'{bench} q={q_val}'):
                doc_ids.append(entry.get('doc_id'))
                if rank_df is None or rank_df.shape[0] == 0:
                    per_q_kendall.append(np.nan)
                    continue

                chatbot_arena_rank = [bench_capability[bench][name] for name in rank_df.columns]
                rerank_result = rerank_matrix_using_df(method, rank_df)
                if rerank_result is None:
                    per_q_kendall.append(np.nan)
                else:
                    try:
                        k_val = pearsonr(rerank_result, chatbot_arena_rank)[0]
                    except Exception:
                        k_val = np.nan
                    per_q_kendall.append(k_val)


            key = f'q_{q_val:.1f}' if q_val is not None else 'original'
            perq_kendall_by_q[key] = per_q_kendall

        # build DataFrame: rows are doc_ids, cols are q values
        perq_df = pd.DataFrame(perq_kendall_by_q)
        perq_df.insert(0, 'doc_id', perq_doc_ids)

        perq_output_dir = f'per_question/{bench}'
        os.makedirs(perq_output_dir, exist_ok=True)
        perq_df.to_csv(f'{perq_output_dir}/kemeny_per_question_kendall_by_q.csv', index=False)
        perq_df.to_excel(f'{perq_output_dir}/kemeny_per_question_kendall_by_q.xlsx', index=False)

        print(f'------finished---{bench}------ saved to {perq_output_dir}')

        # 仅保留 4 位小数的描述性统计输出
        try:
            print(perq_df.describe().round(4).to_markdown())
        except Exception:
            print(perq_df.describe().round(4))