from typing import List, Any, Tuple
from itertools import groupby
import os
import re
from vllm import LLM, SamplingParams
from jinja2 import Template, StrictUndefined
import pandas as pd
from scipy.stats import pearsonr, kendalltau
from utils import (
    load_jsonl_data,
    load_yaml_data,
    # load_json_data,
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


os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

def populate_template(template: str, variables: dict[str, Any]) -> str:
    """
    Populate a Jinja template with variables.
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


def load_llm(model_name_or_path, tokenizer_name_or_path=None, gpu_num=1, lora_model_name_or_path=None):
    """
    Load a VLLM model.
    """
    kw_args = {
        "model": model_name_or_path,
        "tokenizer": tokenizer_name_or_path,
        "tokenizer_mode": "slow",
        "tensor_parallel_size" : gpu_num,
        "enable_lora": bool(lora_model_name_or_path)
    }
    llm = LLM(**kw_args)
    kwargs={
        "n":1,
        "max_tokens":64,
        "top_p":1.0,
        # sampling
        "temperature":0,
        'top_k': 1,
    }
    sampling_params = SamplingParams(**kwargs)
    return llm, sampling_params

def extract_answer(line: str, extract_ABCD=True) -> str:
    """Extracts all possible answer_lst from a line based on known patterns."""
    patterns = [
        re.compile(r"####\s*([\S]+)$"),
        re.compile(r"[Tt]he [Aa]nswer is[:：]?\s*([A-Za-z0-9\u4e00-\u9fa5\.\-\(\)]+)"),
        re.compile(r"答案是[:：]?\s*(.+)"),
        re.compile(r"答案[:：]\s*([A-Za-z0-9\u4e00-\u9fa5．。\-—、《》「」（）()··\"\'·\s]+)"),
        re.compile(r"\\\\?boxed\s*\{\s*([^\{\}]+?)\s*\}"),  # updated
        re.compile(r"\b([A-Da-d]\.\s*[\u4e00-\u9fa5A-Za-z0-9（）《》、“”、·\s]+)"),
    ]
    for pat in patterns:
        for match in pat.findall(line):
            if isinstance(match, tuple):
                answer = match[0].strip()
            else:
                answer = match.strip()
            if extract_ABCD:
                t = re.sub(r'[^ABCD]', '', answer)
                if t:
                    answer = t
            return answer
    return line

def prepare_batch_prompts(prompts_kwargs: List[dict[str, Any]], prompt_template: str, system_prompt='') -> List[str]:
    """
    Prepare a batch of prompts for inference.
    """
    prompts = [populate_template(prompt_template, prompt_kwarg) for prompt_kwarg in prompts_kwargs]
    if system_prompt == '':
        sys_prompt = 'You are a helpful AI assistant.'
    else:
        sys_prompt = system_prompt
    system_prompts = [sys_prompt for _ in prompts]
    message_list = []
    for prompt, sys_prompt in zip(prompts, system_prompts):
        message_list.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ])
    return message_list


def extract_integers(text) -> int:
    """
    正则匹配带或不带千分位逗号的整数
    """
    pattern = r'\b\d{1,3}(?:,\d{3})*\b|\b\d+\b'
    matches = re.findall(pattern, text)

    integers = []
    for match in matches:
        # 去除逗号并转换为整数
        num = int(match.replace(',', ''))
        integers.append(num)

    return integers[0] if len(integers) >= 1 else 0


def post_process(input_lst: List[dict], pred_lst: List[str], answer_lst: List[str]):
    assert len(input_lst) == len(pred_lst)
    assert len(input_lst) == len(answer_lst)

    ans_list = [extract_answer(pred, extract_ABCD=False) for pred in answer_lst]
    ans_list = [extract_integers(ans) for ans in ans_list]

    pred_ans_list = [extract_answer(pred, extract_ABCD=False) for pred in pred_lst]
    pred_ans_list = [extract_integers(pred) for pred in pred_ans_list]

    for input, pred, pred_ans, raw_ans, ans in zip(input_lst, pred_lst, pred_ans_list, answer_lst, ans_list):
        input['pred'] = pred
        input['pred_ans'] = pred_ans
        input['raw_answer'] = raw_ans
        input['answer'] = ans
        input['correct'] = ans == pred_ans
    return input_lst

def calc_corr(model_rank: dict, human_rank: dict, correlation='pearson') -> float:
    model_rank_df = pd.Series(model_rank)
    human_rank_df = pd.Series(human_rank)
    model_human_rank = pd.concat([model_rank_df, human_rank_df], axis=1).T
    model_r = model_human_rank.iloc[0]
    human_r = model_human_rank.iloc[1]
    if correlation == 'pearson':
        return pearsonr(model_r, human_r)[0]
    elif correlation == 'kendall':
        return kendalltau(model_r, human_r)[0]
    else:
        raise ValueError(f"Invalid correlation metric: {correlation}")

def calculate_accuracy(data: List[dict]) -> Tuple[float, int, int]:
    """Calculate accuracy from data with 'correct' field."""
    correct = sum(1 for item in data if item.get('correct', False))
    total = len(data)
    return (correct / total * 100, correct, total)

if __name__ == '__main__':
    resp_models = ['gpt-4o-2024-11-20', 'gpt-4o-2024-05-13', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229', 'gpt-4o-2024-08-06']
    PROMPT_TEMPLATES = load_yaml_data('extract_ans_prompts.yaml')
    model_name_or_path = '/home/share/models/modelscope/Qwen/Qwen2.5-32B-Instruct'
    accs = {}
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
    for resp_model in resp_models:
        data = load_jsonl_data(f'answer/gsm8k/{resp_model}/inference_results.jsonl')
        accuracy, correct, total = calculate_accuracy(data)
        accs[resp_model] = accuracy

    accs = sorted(accs.items(), key=lambda x: x[1], reverse=True)
    print(f"accs: {accs}")
    groups = [(score, list(items)) for score, items in groupby(accs, key=lambda x: x[1])]

    accs_rank = {}
    current_rank = 1
    for score, items in groups:
        for model, _ in items:
            accs_rank[model] = current_rank
        current_rank += 1
    print(f"accs_rank: {accs_rank}")
    pearson = calc_corr(accs_rank, bench_capability['gsm8k'], 'pearson')
    kendall = calc_corr(accs_rank, bench_capability['gsm8k'], 'kendall')
    print(f'Pearson: {pearson}, Kendall: {kendall}')

    # model, sampling_params = load_llm(model_name_or_path, gpu_num=4)
    # for resp_model in resp_models:
    #     datas = load_jsonl_data(f'outputs/gsm8k/{resp_model}/inference_results.jsonl')
    #     prompts = []
    #     inputs, answer_lst = [], []
    #     for data in datas:
    #         prompts.append({
    #             'question': data['question'],
    #             'response_text': data['response'],
    #         })
    #         inputs.append(
    #         {
    #             'question': data['question'],
    #             'resp': data['response'],
    #         })
    #         answer_lst.append(data['answer'])
    #     all_prompts = prepare_batch_prompts(
    #         prompts,
    #         prompt_template=PROMPT_TEMPLATES['gsm8k_extraction']['user_template'],
    #         system_prompt=''
    #     )
    #     # outputs_t = model.chat(all_prompts, sampling_params, use_tqdm=True)

    #     pred_lst = []
    #     for o_t in outputs_t:
    #         pred_lst.append(o_t.outputs[0].text)
    #     results = post_process(inputs, pred_lst, answer_lst)
    #     save_jsonl_data(results, f'answer/gsm8k/{resp_model}/inference_results.jsonl')
