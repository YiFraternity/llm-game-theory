import re
import json
from typing import List, Tuple, Dict

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


def extract_answers(line: str, extract_ABCD=True) -> str:
    """Extracts all possible answers from a line based on known patterns."""
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

def process_file(filename: str, model_id_name: Dict[str, str]) -> List[Tuple[int, str, List[str]]]:
    results = []
    datas = load_json_data(filename)
    for data in datas:
        question = data.get('question') or data.get('prompt', '')
        id = data.get('id') if data.get('id') is not None else data.get('doc_id')
        answer = data.get('target', None)
        q_id = {
            'id': id,
            'question': question,
        }
        if answer:
            answer = re.sub(r'[^ABCD]', '', answer)
            q_id['answer'] = answer
        model_info = []
        resps = data.get('resps', [])
        models = data.get('models', [])
        if not models:
            models = [model_id_name[str(i+1)] for i in range(len(resps))]
        for resp, model in zip(resps, models):
            model_resp = extract_answers(resp)
            model_info.append({
                'model': model,
                'resp': resp,
                'answer': model_resp,
            })
        q_id['model_info'] = model_info
        results.append(q_id)
    return results


if __name__ == '__main__':
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
    for bench in benchmarks:
        results = process_file(f'outputs/{bench}/merged_results.json', model_id_name[bench])
        with open(f'outputs/{bench}/answers.json', 'w') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))

