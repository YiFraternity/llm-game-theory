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
from Q1_each_model import get_rank_matrix
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
    GEN_OVERALL_RANKING,
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
    print(each_question_ranks_df.describe())
    total_rerank_ranking = np.mean(total_ranks_avgs, axis=0)
    print(total_rerank_ranking)
    print(chatbot_arena_rank)
    total_pearson_correlation = np.corrcoef(chatbot_arena_rank, total_rerank_ranking)[0][1]
    print(total_pearson_correlation)
    return pearson_correlations, total_pearson_correlation


if __name__ == "__main__":
    benchmarks = ["Q4/chinese"]
    bench_capability = {
        'mmlu': OVERALL_RANKING,
        'mbpp': CREATING_WRITING,
        'ifeval': INSTRUCTION_RANKING,
        'writingbench': CODEING,
        'ceval': CHINESE_RANKING,
        'gpqa': OVERALL_RANKING,
        'gsm8k': MATHEMATIC,
        'Q4/math': MATHEMATIC,
        'Q4/overall': GEN_OVERALL_RANKING,
        'Q4/chinese': CHINESE_RANKING,
    }
    model_id_name = {
        'mmlu': MMLU_MODEL_ID_NAME,
        'mbpp': MBPP_MODEL_ID_NAME,
        'ifeval': IFEVAL_MODEL_ID_NAME,
        'writingbench': WRITING_BENCH_MODEL_ID_NAME,
        'ceval': CEVAL_MODEL_ID_NAME,
        'gpqa': GPQA_MODEL_ID_NAME,
        'gsm8k': GSM8K_MODEL_ID_NAME,
        'Q4/math': MBPP_MODEL_ID_NAME,
        'Q4/overall': MBPP_MODEL_ID_NAME,
        'Q4/chinese': MBPP_MODEL_ID_NAME,
    }
    methods = [
        'average',
        'dodgeson',
        'copeland',
        'borda',
        'irv',
        'spearman',
        'kemeny_young',
        'kendall',
    ]

    # methods = ['kemeny_young']
    # 模型排序与人类判断的一致性，与不同重排序（Rerank）方法与人类一致性之间的kendall和pearson相关系数对比
    model_macro_kendall, model_macro_pearson = {method: {} for method in methods}, {method: {} for method in methods}
    for bench in benchmarks:
        input_file = f'outputs/{bench}/input/ranks.json'
        data = load_json_data(input_file)
        rank_matrixs = get_rank_matrix(data, model_id_name[bench], bench_capability[bench])

        model_micro_kendall, model_micro_pearson = {method: [] for method in methods}, {method: [] for method in methods}
        for method in methods:
            all_question_rank = []
            for rank_df in tqdm(rank_matrixs):   # 每道题目
                chatbot_arena_rank = [bench_capability[bench][name] for name in rank_df.columns]
                rerank_result = rerank_matrix_using_df(method, rank_df)
                if rerank_result is None:
                    model_micro_kendall[method].append(np.nan)
                    model_micro_pearson[method].append(np.nan)
                    continue
                model_micro_kendall[method].append(kendalltau(rerank_result, chatbot_arena_rank)[0])
                model_micro_pearson[method].append(pearsonr(rerank_result, chatbot_arena_rank)[0])
                all_question_rank.append(rerank_result)
            all_question_rank = [_ for _ in all_question_rank if _]
            all_quest_avg = np.mean(all_question_rank, axis=0)
            model_macro_kendall[method][bench] = kendalltau(all_quest_avg, chatbot_arena_rank)[0]
            model_macro_pearson[method][bench] = pearsonr(all_quest_avg, chatbot_arena_rank)[0]
        # model_micro_pearson_df = pd.DataFrame(
        #     dict([(k, pd.Series(v)) for k, v in model_micro_pearson.items()])
        # )
        # model_micro_kendall_df = pd.DataFrame(
        #     dict([(k, pd.Series(v)) for k, v in model_micro_kendall.items()])
        # )
        model_micro_pearson_df = pd.DataFrame(model_micro_pearson)
        model_micro_kendall_df = pd.DataFrame(model_micro_kendall)
        kendall_output_dir = f'kendall/{bench}'
        pearson_output_dir = f'pearson/{bench}'
        os.makedirs(pearson_output_dir, exist_ok=True)
        model_micro_pearson_df.to_excel(f'{pearson_output_dir}/rerank_pearson_corr_micro.xlsx')
        model_micro_pearson_df.describe().to_excel(f'{pearson_output_dir}/rerank_pearson_corr_micro_describe.xlsx')
        os.makedirs(kendall_output_dir, exist_ok=True)
        model_micro_kendall_df.to_excel(f'{kendall_output_dir}/rerank_kendall_corr_micro.xlsx')
        model_micro_kendall_df.describe().to_excel(f'{kendall_output_dir}/rerank_kendall_corr_micro_describe.xlsx')
        row_data = {method: values[bench] for method, values in model_macro_pearson.items()}
        model_macro_pearson_df = pd.DataFrame([row_data], index=[bench])
        model_macro_pearson_df.to_excel(f'{pearson_output_dir}/rerank_pearson_corr_macro.xlsx')

        row_data = {method: values[bench] for method, values in model_macro_kendall.items()}
        model_macro_kendall_df = pd.DataFrame([row_data], index=[bench])
        model_macro_kendall_df.to_excel(f'{kendall_output_dir}/rerank_kendall_corr_macro.xlsx')
        print(f'------finished---{bench}------')

    # for method in methods:
    #     model_micro_pearson_df = pd.DataFrame(model_macro_pearson[method], index=[0])
    #     model_micro_kendall_df = pd.DataFrame(model_macro_kendall[method], index=[0])
    #     kendall_output_dir = f'kendall'
    #     pearson_output_dir = f'pearson'
    #     os.makedirs(pearson_output_dir, exist_ok=True)
    #     model_micro_pearson_df.to_excel(f'{pearson_output_dir}/rerank_{method}_pearson_corr_macro.xlsx')
    #     os.makedirs(kendall_output_dir, exist_ok=True)
    #     model_micro_kendall_df.to_excel(f'{kendall_output_dir}/rerank_{method}_kendall_corr_macro.xlsx')

    # 对比同rerank方法，在不同的任务上的表现
    """
    method = 'kendall'
    task_pearson_list, task_macro_corr = {}, {}
    for bench in benchmarks:
        input_file = f'outputs/{bench}/input/ranks.json'
        data = load_json_data(input_file)
        rank_matrix = get_rank_matrix(data, model_id_name[bench], bench_capability[bench])
        pearson_list, all_question_rank = [], []
        for rank_df in tqdm(rank_matrix):   # 每道题目
            chatbot_arena_rank = [bench_capability[bench][name] for name in rank_df.columns]
            pearson_corr, rerank_result = rerank_matrix_using_df(method, rank_df, chatbot_arena_rank)
            pearson_list.append(pearson_corr)
            all_question_rank.append(rerank_result)
        all_question_rank = [_ for _ in all_question_rank if _]
        all_quest_avg = np.mean(all_question_rank, axis=0)
        overall_pearson_corr = np.corrcoef(all_quest_avg, chatbot_arena_rank)[0][1]
        task_pearson_list[bench] = pearson_list
        task_macro_corr[bench] = overall_pearson_corr
    task_each_quest_corr_df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in task_pearson_list.items()])
    )
    output_dir = f'kendall'
    os.makedirs(output_dir, exist_ok=True)
    task_each_quest_corr_df.to_excel(f'{output_dir}/kendall_pearson_corr_micro.xlsx')
    task_each_quest_corr_df.describe().to_excel(f'{output_dir}/kendall_pearson_corr_micro_describe.xlsx')
    task_all_macro_corr_df = pd.DataFrame(task_macro_corr, index=[0])
    task_all_macro_corr_df.to_excel(f'{output_dir}/kendall_pearson_corr_macro.xlsx')
    print(f'------finished---{bench}')
    """

    # for methon in methods:
    #     pearson_list, total_pearson_list = [], []
    #     for benchmark in benchmarks:
    #         input_file = f'outputs/{benchmark}_ranks.xlsx'
    #         chatbot_arena_rank = bench_capability[benchmark]
    #         pearson_correlations, total_pearson_correlation = main(method, input_file, chatbot_arena_rank)
    #         pearson_list.append(pearson_correlations)
    #         total_pearson_list.append(total_pearson_correlation)

    #     max_len = max(len(p) for p in pearson_list)
    #     padded = [np.pad(np.array(p), (0, max_len - len(p)), constant_values=np.nan) for p in pearson_list]
    #     pearson_array = np.column_stack(padded)

    #     nan_rows = np.full((5, pearson_array.shape[1]), np.nan)
    #     total_row = np.array(total_pearson_list).reshape(1, -1)
    #     final_array = np.vstack([pearson_array, nan_rows, total_row])

    #     df = pd.DataFrame(final_array, columns=[f'{b}_pearson_correlation' for b in benchmarks])
    #     df.to_excel(f'pearson_correl/{method}_pearson_correlations.xlsx', index=False)
    #     print(f'-------------finish {method}-------------------')

