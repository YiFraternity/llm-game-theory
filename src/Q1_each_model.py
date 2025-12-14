import os
import copy
import json
from typing import Dict, List, Any
import numpy as np
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
)



def calc_one_model_corrlection(
    data: List[Dict[str, Any]],
    capability: Dict[str, int],
    model_id_name: Dict[str, str],
    correlation: str = 'pearson'
) -> Dict[str, List[int]]:
    """
    计算每个模型的pearson相关系数
    """
    if not data:
        return {}
    models = data[0]['models']
    model_rankings = {model: [] for model in models}
    model_macro_corr = {model: 0 for model in models}
    model_micro_corrs = {model: [] for model in models}
    human_rank = pd.Series(capability)
    for item in data:
        cur_models = item['models']
        assert set(cur_models) == set(models)
        for model, resp in zip(cur_models, item['resps']):
            if len(resp) < 6:
                model_rankings[model].append({})
                continue
            resp_id = resp[:6]
            resp_id = list(resp_id)
            resp_model_rank = {model_id_name[str(i+1)]: int(c) for i, c in enumerate(resp_id)}
            corr_df = pd.DataFrame([resp_model_rank, capability])
            if correlation == 'pearson':
                corr = pearsonr(corr_df.iloc[0], corr_df.iloc[1])[0]
            elif correlation == 'kendall':
                corr = kendalltau(corr_df.iloc[0], corr_df.iloc[1])[0]
            else:
                raise ValueError(f"Invalid correlation metric: {correlation}")
            model_micro_corrs[model].append(corr)
            model_rankings[model].append(resp_model_rank)
    for model in models:
        model_rank_df = pd.DataFrame(model_rankings[model])
        model_rank_avg = model_rank_df.mean(axis=0)
        rank_avg = pd.concat([model_rank_avg, human_rank], axis=1).T
        if correlation == 'pearson':
            model_macro_corr[model] = pearsonr(rank_avg.iloc[0], rank_avg.iloc[1])[0]
        elif correlation == 'kendall':
            model_macro_corr[model] = kendalltau(rank_avg.iloc[0], rank_avg.iloc[1])[0]
        else:
            raise ValueError(f"Invalid correlation metric: {correlation}")
    return model_macro_corr, model_micro_corrs

if __name__ == "__main__":
    benchmarks = ['ceval', 'ifeval', 'mbpp', 'writingbench', 'gpqa', 'gsm8k', 'mmlu']
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
    correlations = ['kendall', 'pearson']
    rename = {
        'chatgpt-4o-latest-20241120': 'gpt-4o-2024-11-20',
        'gpt-4o-2024-05-13': 'gpt-4o-2024-05-13',
        'claude-3-5-sonnet-20241022': 'claude-3-5-sonnet-20241022',
        'claude-3-5-haiku-20241022': 'claude-3-5-haiku-20241022',
        'claude-3-opus-20240229': 'claude-3-opus-20240229',
        'gpt-4o-2024-08-06': 'gpt-4o-2024-08-06',
    }
    model_order = [
        'gpt-4o-2024-11-20',
        'claude-3-5-sonnet-20241022',
        'gpt-4o-2024-05-13',
        'gpt-4o-2024-08-06',
        'claude-3-opus-20240229',
        'claude-3-5-haiku-20241022',
    ]
    for correlation in correlations:
        for bench in benchmarks:
            input_file = f'outputs/{bench}/input/ranks.json'
            data = load_json_data(input_file)

            model_macro_corr, model_micro_corrs = calc_one_model_corrlection(
                data, bench_capability[bench],
                model_id_name[bench],
                correlation
            )

            os.makedirs(f"outputs/{bench}", exist_ok=True)

            max_len = max(len(corrs) for corrs in model_micro_corrs.values())
            for model in model_micro_corrs:
                corrs = model_micro_corrs[model]
                if len(corrs) < max_len:
                    model_micro_corrs[model] = corrs + [np.nan] * (max_len - len(corrs))

            model_micro_corrs_df = pd.DataFrame(model_micro_corrs)
            model_micro_corrs_df = model_micro_corrs_df.rename(columns=rename)
            model_micro_corrs_df = model_micro_corrs_df.reindex(model_order, axis=1)
            output_dir = f'{correlation}/{bench}'
            os.makedirs(output_dir, exist_ok=True)
            model_micro_corrs_df.to_excel(f"{output_dir}/one_model_{correlation}_corr_micro.xlsx", index=False)
            model_micro_corrs_df.describe().to_excel(f"{output_dir}/one_model_{correlation}_corr_micro_description.xlsx")

            model_macro_corrs_df = pd.DataFrame(model_macro_corr, index=['corr'])
            model_macro_corrs_df = model_macro_corrs_df.rename(columns=rename)
            model_macro_corrs_df = model_macro_corrs_df.reindex(model_order, axis=1)
            model_macro_corrs_df.to_excel(f"{output_dir}/one_model_{correlation}_corr_macro.xlsx", index=False)
            print(f"Output saved to {output_dir}")

