import os
import copy
from typing import Dict, List, Any
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as stats
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
    Rerank,
)


def get_reps_model_ranks(
    data: List[Dict[str, Any]],
    model_id_name: Dict[str, str],
) -> List[Dict[str, Dict[str, int]]]:
    """
    针对每一道题，得到一个字典，Key是模型名，Value是一个每一个model的排名。
    Returns:
        4o-11-26: {4o-05-10: rank, 4o-11-26: rank, ...}
    """
    if not data:
        return []
    models = data[0]['models']
    all_questions = []
    for item in data:
        question = {}
        cur_models = item['models']
        assert set(cur_models) == set(models)
        for model, resp in zip(cur_models, item['resps']):
            if len(resp) < 6:
                continue
            resp_id = resp[:6]
            resp_id = list(resp_id)
            resp_model_rank = {model_id_name[str(i+1)]: int(c) for i, c in enumerate(resp_id)}
            question[model] = resp_model_rank
        all_questions.append(question)
    return all_questions


def calc_self_other_rank(model_rank: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """
    计算每道题目 中 每个模型的self-other rank
    Args:
        model_rank: 每道题目对应的模型排名
            Example: {'gpt-4o-2024-11-26': {'gpt-4o-2024-05-10': 1, 'claude-3-opus-20240229': 2, ...}, ...}
    Returns:
        DataFrame: 包含每个模型的self_rank和other_rank
            Example:
           self_rank  {'gpt-4o-2024-11-26': 1, 'gpt-4o-2024-05-10': 2.5}
           other_rank {'gpt-4o-2024-11-26': 2, 'gpt-4o-2024-05-10': 2.5}
    """
    self_ranks = {}   # 存储每个模型给自己的评分
    other_ranks = {}  # 存储每个模型收到其他模型的评分
    for model_name, ranks in model_rank.items():
        self_ranks[model_name] = ranks.get(model_name, np.nan)
        for o_name, o_rank in ranks.items():
            if o_name not in other_ranks:
                other_ranks[o_name] = []
            other_ranks[o_name].append({model_name: o_rank})
    other_avg_ranks = {}
    for name, ranks in other_ranks.items():
        if not ranks:
            continue
        avg_rank = np.mean([list(rank.values())[0] for rank in ranks])
        other_avg_ranks[name] = avg_rank
    return pd.DataFrame({'self_rank': self_ranks, 'other_rank': other_avg_ranks}).T


def calc_self_other_rank_corr(
    model_ranks: List[Dict[str, Dict[str, int]]],
    human_rank: Dict[str, int],
    correlation: str = 'pearson'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算每个模型的self-other rank 与Human rank的Pearson相关系数
    Args:
        model_ranks: 所有题目对应的模型排名
            Example: {'gpt-4o-2024-11-26': {'gpt-4o-2024-05-10': 1, 'claude-3-opus-20240229': 2, ...}, ...}
        human_rank: 人类排名
            Example: {'gpt-4o-2024-11-26': 1, 'gpt-4o-2024-05-10': 2, ...}
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - micro_df: 每个模型的self-other rank 与Human rank的Pearson相关系数
            - macro_df: 所有模型的self-other rank 与Human rank的Pearson相关系数
    """
    all_self_ranks = []
    all_other_ranks = []
    micro_self_corr = []
    micro_other_corr = []
    human_rank_df = pd.Series(human_rank)

    def calc_corr(model_rank: pd.Series):
        """
        计算Pearson或者Kendall相关系数
        """
        rank_df = pd.concat([model_rank, human_rank_df], axis=1).T
        self_rank = rank_df.iloc[0]  # 第一行
        human_rank = rank_df.iloc[1]  # 第二行
        if len(self_rank) != len(human_rank):
            return np.nan
        if correlation == 'pearson':
            return stats.pearsonr(self_rank, human_rank)[0]
        elif correlation == 'kendall':
            return stats.kendalltau(self_rank, human_rank)[0]
        else:
            raise ValueError(f"Invalid correlation metric: {correlation}")

    for question in model_ranks:
        self_other_rank_df = calc_self_other_rank(question)
        self_rank_df = self_other_rank_df.loc['self_rank']
        other_rank_df = self_other_rank_df.loc['other_rank']
        all_self_ranks.append(self_rank_df)
        all_other_ranks.append(other_rank_df)

        micro_self_corr.append(calc_corr(self_rank_df))
        micro_other_corr.append(calc_corr(other_rank_df))

    all_self_ranks_df = pd.concat(all_self_ranks, axis=1).T
    all_other_ranks_df = pd.concat(all_other_ranks, axis=1).T
    avg_self_ranks_df = all_self_ranks_df.mean(axis=0)
    avg_other_ranks_df = all_other_ranks_df.mean(axis=0)
    macro_self_corr = calc_corr(avg_self_ranks_df)
    macro_other_corr = calc_corr(avg_other_ranks_df)
    micro_df = pd.DataFrame({'SIE': micro_self_corr, 'others': micro_other_corr})
    macro_df = pd.DataFrame({'SIE': [macro_self_corr], 'others': [macro_other_corr]}, index=['macro'])
    return micro_df, macro_df


def calc_avg_self_other_rank(model_ranks: List[Dict[str, Dict[str, int]]]) -> pd.DataFrame:
    """
    计算所有题目每个模型的self-other rank的平均值
    Returns:
        DataFrame: 包含每个模型的平均self_rank和other_rank
    """
    dfs = [calc_self_other_rank(question).T for question in model_ranks]
    if not dfs:
        return pd.DataFrame()
    avg_df = pd.concat(dfs).groupby(level=0).mean()
    return avg_df.T

def calc_corr(model_rank: dict, human_rank: dict, correlation='pearson') -> float:
    model_rank_df = pd.Series(model_rank)
    human_rank_df = pd.Series(human_rank)
    model_human_rank = pd.concat([model_rank_df, human_rank_df], axis=1).T
    model_r = model_human_rank.iloc[0]
    human_r = model_human_rank.iloc[1]
    if correlation == 'pearson':
        return stats.pearsonr(model_r, human_r)[0]
    elif correlation == 'kendall':
        return stats.kendalltau(model_r, human_r)[0]
    else:
        raise ValueError(f"Invalid correlation metric: {correlation}")

def get_rank_matrix(data: List[Dict[str, Any]], model_id_name: Dict[str, str], capability: Dict[str, int]) -> List[pd.DataFrame]:
    """
    应该是针对每一道题，得到一个排名矩阵。
    最终矩阵的列是capability中模型的排名。
    """
    if not data:
        return []
    models = data[0]['models']
    rank_matrix = []
    model_rankings = sorted(capability.items(), key=lambda x: x[1])
    model_rankings = [x[0] for x in model_rankings]
    for item in data:
        matrix = {}
        cur_models = item['models']
        assert set(cur_models) == set(models)
        for model, resp in zip(item['models'], item['resps']):
            if len(resp) < 6:
                continue
            resp_id = resp[:6]
            resp_id = list(resp_id)
            resp_model_rank = {model_id_name[str(i+1)]: int(c) for i, c in enumerate(resp_id)}
            matrix[model] = resp_model_rank
        matrix_df = pd.DataFrame.from_dict(matrix, orient='index')
        rank_matrix.append(matrix_df)
    return rank_matrix


def calc_self_rerank_metric(
    rank_df: pd.DataFrame,
    rerank_method: str,
    model_name: str
) -> tuple[float, float]:
    """
    得到每道题目指定模型的自我排名，以及其他模型的打分情况对该模型提供合适位次
    Args:
        rank_df: DataFrame
        rerank_method: rerank方法, kendall, borda, average, spearman, kemeny_young, irv
        model_name: 计算指定模型的rank
    Returns:
        self_eval: 每个模型的自我排名
        peer_eval: 其他模型对该模型的平均排名
        sie_eval: 包含自我排名和其他模型对该模型的，重排后的排名
        sfe_eval: 移除自我排名，其他模型对该模型的重排后的排名
    """
    model_column_index = rank_df.columns.get_loc(model_name)
    if model_name in rank_df.index:
        # 取出行列均为model的df值
        self_eval = rank_df.loc[model_name, model_name]
        # 取出行不为model的df值
        peer_rank_df = rank_df.loc[rank_df.index != model_name]
        peer_ranks = peer_rank_df.values.tolist()
    else:
        self_eval = np.nan
        peer_ranks = rank_df.values.tolist()

    peer_eval = np.nanmean(peer_ranks, axis=0)[model_column_index]

    all_ranks = rank_df.values.tolist()
    sie_rerank = Rerank(all_ranks)
    sie_rerank_ranking = sie_rerank.rerank_method(rerank_method)
    assert len(rank_df.columns) == len(sie_rerank_ranking)
    sie_eval = sie_rerank_ranking[model_column_index]

    sfe_ranks = copy.deepcopy(all_ranks)
    try:
        model_name_values = sfe_ranks[rank_df.index.get_loc(model_name)]
        model_name_column_index = rank_df.columns.get_loc(model_name)
        model_name_values[model_name_column_index] = peer_eval
        origin_values = copy.deepcopy(model_name_values)
        indexed_list = sorted([(value, i) for i, value in enumerate(origin_values)])
        for i, (_, origin_values) in enumerate(indexed_list):
            model_name_values[origin_values] = i + 1
    except:
        sfe_ranks = rank_df.values.tolist()

    sfe_rerank = Rerank(sfe_ranks)
    sfe_rerank_ranking = sfe_rerank.rerank_method(rerank_method)
    assert len(rank_df.columns) == len(sfe_rerank_ranking)
    sfe_eval = sfe_rerank_ranking[model_column_index]

    return self_eval, peer_eval, sie_eval, sfe_eval


if __name__ == "__main__":
    benchmarks = ['ceval', 'ifeval', 'mbpp', 'writingbench', 'gsm8k', 'gpqa', 'mmlu']
    bench_capability = {
        'ceval': CHINESE_RANKING,
        'ifeval': INSTRUCTION_RANKING,
        'mbpp': CREATING_WRITING,
        'writingbench': CODEING,
        'gsm8k': MATHEMATIC,
        'gpqa': OVERALL_RANKING,
        'mmlu': OVERALL_RANKING,
    }
    model_id_name = {
        'ceval': CEVAL_MODEL_ID_NAME,
        'ifeval': IFEVAL_MODEL_ID_NAME,
        'mbpp': MBPP_MODEL_ID_NAME,
        'writingbench': WRITING_BENCH_MODEL_ID_NAME,
        'gsm8k': GSM8K_MODEL_ID_NAME,
        'gpqa': GPQA_MODEL_ID_NAME,
        'mmlu': MMLU_MODEL_ID_NAME,
    }
    rename = {
        'chatgpt-4o-latest-20241120': 'gpt-4o-2024-11-20',
        'gpt-4o-2024-05-13': 'gpt-4o-2024-05-13',
        'claude-3-5-sonnet-20241022': 'claude-3-5-sonnet-20241022',
        'claude-3-5-haiku-20241022': 'claude-3-5-haiku-20241022',
        'claude-3-opus-20240229': 'claude-3-opus-20240229',
        'gpt-4o-2024-08-06': 'gpt-4o-2024-08-06',
    }
    methods = [
        'dodgeson',
        'average',
        'copeland',
        'condorcet',
        'borda',
        'irv',
        'spearman',
        'kemeny_young',
        'kendall',
    ]
    correlations = ['pearson', 'kendall']

    # 我要得到一个这样的结果：
    # 针对每一种重排结果，用于记录模型对自己在数据集中的排名Self和重排后模型的合理排名Rerank的对比
    # 每个Excel有8列，其中每列表示一个模型
    # 每个Excel有Benchmark数量*2 行，针对每个Benchmark，记录Self和Rerank的对比
    method = 'kemeny_young'
    avg_benchmark_rank_dicts = {bench: {} for bench in benchmarks}
    benchmark_pearson_dicts = {bench: {'SE': [], 'PE': [], 'SIE': [], 'SFE': []} for bench in benchmarks}
    for bench in benchmarks:
        input_file = f'outputs/{bench}/input/ranks.json'
        data = load_json_data(input_file)
        ranks = get_rank_matrix(data, model_id_name[bench], bench_capability[bench])
        models = ranks[0].columns
        each_model_rank_dicts = { # 自我排名和重排中每道题目每个模型的排名
            'SE': {model: [] for model in models},   # 自我排名
            'PE': {model: [] for model in models},   # Peer Models的排名
            'SIE': {model: [] for model in models},  # 包含自我评分后的排名，如果 SIE 显著偏离 SFE，说明模型的自评分数污染了聚合结果，博弈论机制抗干扰能力弱。
            'SFE': {model: [] for model in models},  # 消除自我偏见后的排名，代表了博弈论机制在“去自我影响”条件下的最理想效果。
        }

        for rank_df in tqdm(ranks):
            _model_self_eval_dict_ = {model: np.nan for model in models}
            _model_peer_eval_dict_ = {model: np.nan for model in models}
            _model_sie_eval_dict_ = {model: np.nan for model in models}
            _model_sfe_eval_dict_ = {model: np.nan for model in models}
            for model in models:
                self_eval, peer_eval, sie_eval, sfe_eval = calc_self_rerank_metric(
                    rank_df=rank_df,
                    rerank_method=method,
                    model_name=model
                )
                _model_self_eval_dict_[model] = self_eval
                _model_peer_eval_dict_[model] = peer_eval
                _model_sie_eval_dict_[model] = sie_eval
                _model_sfe_eval_dict_[model] = sfe_eval

                each_model_rank_dicts['SE'][model].append(self_eval)
                each_model_rank_dicts['PE'][model].append(peer_eval)
                each_model_rank_dicts['SIE'][model].append(sie_eval)
                each_model_rank_dicts['SFE'][model].append(sfe_eval)
            if any(isinstance(v, float) and np.isnan(v) for v in _model_self_eval_dict_.values()):
                self_pearson = np.nan
            else:
                self_pearson = calc_corr(
                    _model_self_eval_dict_,
                    bench_capability[bench],
                    correlation='pearson')
            peer_pearson = calc_corr(
                _model_peer_eval_dict_,
                bench_capability[bench],
                correlation='pearson')
            sie_pearson = calc_corr(
                _model_sie_eval_dict_,
                bench_capability[bench],
                correlation='pearson')
            sfe_pearson = calc_corr(
                _model_sfe_eval_dict_,
                bench_capability[bench],
                correlation='pearson')

            benchmark_pearson_dicts[bench]['SE'].append(self_pearson)
            benchmark_pearson_dicts[bench]['PE'].append(peer_pearson)
            benchmark_pearson_dicts[bench]['SIE'].append(sie_pearson)
            benchmark_pearson_dicts[bench]['SFE'].append(sfe_pearson)

        avg_benchmark_rank_dicts[bench] = { # 自我排名和重排中每道题目每个模型的平均排名
            'SE': {model: np.nanmean(each_model_rank_dicts['SE'][model]) for model in models},
            'PE': {model: np.nanmean(each_model_rank_dicts['PE'][model]) for model in models},
            'SIE': {model: np.nanmean(each_model_rank_dicts['SIE'][model]) for model in models},
            'SFE': {model: np.nanmean(each_model_rank_dicts['SFE'][model]) for model in models},
        }

    # 将benchmark_pearson_dicts字典转换为DataFrame
    # 找出最大长度，用于对齐
    max_len = max(len(values['SIE']) for values in benchmark_pearson_dicts.values())
    rows = []
    for i in range(max_len):
        row = {}
        for bench, data in benchmark_pearson_dicts.items():
            for key in ['SE', 'PE', 'SIE', 'SFE']:
                col_name = f"{bench}_{key}"
                try:
                    row[col_name] = data[key][i]
                except IndexError:
                    row[col_name] = float('nan')
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel('self-rerank/pearson.xlsx', index=True, header=True)

    # 将平均排名字典转换为DataFrame
    for dataset in avg_benchmark_rank_dicts.values():
        for method in dataset.values():
            to_delete = []
            to_add = {}
            for key in method.keys():
                if key in rename:
                    new_key = rename[key]
                    to_delete.append(key)
                    to_add[new_key] = method[key]
            for key in to_delete:
                del method[key]
            method.update(to_add)

    # 构建 DataFrame
    rows = []
    for dataset, methods in avg_benchmark_rank_dicts.items():
        for method, models in methods.items():
            row = {'dataset': dataset, 'method': method}
            row.update(models)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index(['dataset', 'method'], inplace=True)
    df.to_excel('self-rerank/avg_self-rerank.xlsx', index=True, header=True)




    """
    avg_self_other_rank = {}
    micro_pearson = {}
    macro_pearson = {}
    for bench in benchmarks:
        input_file = f'outputs/{bench}/input/ranks.json'
        data = load_json_data(input_file)
        self_other_rank = get_reps_model_ranks(data, model_id_name[bench])
        bench_df = calc_avg_self_other_rank(self_other_rank)
        bench_df.rename(columns=rename, inplace=True)
        bench_df['benchmark'] = bench
        avg_self_other_rank[bench] = bench_df

        micro_df, macro_df = calc_self_other_rank_pearson(self_other_rank, bench_capability[bench])
        micro_df['benchmark'] = bench
        macro_df['benchmark'] = bench
        micro_pearson[bench] = micro_df
        macro_pearson[bench] = macro_df

    avg_self_other_rank_df = pd.concat(avg_self_other_rank.values())
    avg_self_other_rank_df.set_index('benchmark', append=True, inplace=True)
    avg_self_other_rank_df = avg_self_other_rank_df.reorder_levels(['benchmark', None])

    os.makedirs('self_others', exist_ok=True)
    avg_self_other_rank_df.to_excel('self_others/avg_self_other_rank.xlsx', index=True, header=True)

    micro_pearson_df = pd.concat(micro_pearson.values(), axis=1)
    macro_pearson_df = pd.concat(macro_pearson.values(), axis=1)
    micro_pearson_df.to_excel('self_others/micro_pearson.xlsx', index=True, header=True)
    macro_pearson_df.to_excel('self_others/macro_pearson.xlsx', index=True, header=True)
    """


