"""
处理多个AI模型的排名结果，把不同模型对同一问题的回答整理到一起，方便比较。

## 输入
1. **模型输出文件**：
   - 在`lm-evaluation-harness/output/`目录下找各个模型的`_rank.jsonl`文件
   - 每个文件包含模型对问题的回答和排名

2. **模型对应关系文件**：
   - 每个测试目录下的`对应关系.json`文件
   - 记录模型名称和编号的对应关系

## 输出
- 生成`ranks.json`文件
- 包含所有模型对每个问题的回答排名

```json
{
  "问题ID": "123",
  "问题内容": "问题文本",
  "正确答案": "目标答案",
  "各模型排名": ["156234", "234156", ...],
  "模型名称": ["gpt-4", "claude-3", ...],
  "选项": ["A", "B", "C", ...]  // 如果有的话
}
```

## 错误处理
- 自动跳过不认识的模型
- 检查不同模型回答的是不是同一个问题
- 能处理格式不规范的排名文字
"""

import os
import json
import re
from enum import Enum
from typing import List, Tuple, Dict


class ModelName(Enum):
    # 0. chatgpt-4o-latest = chatgpt-4o-latest-20241120
    # 1. gpt-4o-2024-05-13 = gpt-4o-2024-05-13
    # 2. claude-3-5-sonnet = claude-3-5-sonnet-20241022
    # 3. claude-3-5-haiku = claude-3-5-haiku-20241022
    # 4. claude-3-opus = claude-3-opus-20240229
    # 5. gpt-4o = gpt-4o-2024-08-06
    CHATGPT_4O_LATEST = "chatgpt-4o-latest-20241120"
    GPT_4O_OLD = "gpt-4o-2024-05-13"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    GPT_4O = "gpt-4o-2024-08-06"
    ERROR = "error"

    @staticmethod
    def get_model_name(model_name):
        if model_name in ["chatgpt-4o-latest", "chatgpt-4o-latest-20241120", "gpt-4o-2024-11-20"]:
            return ModelName.CHATGPT_4O_LATEST
        elif model_name in ["gpt-4o-2024-05-13", "chatgpt-4o-old"]:
            return ModelName.GPT_4O_OLD
        elif model_name in ["claude-3-5-sonnet", "claude-3-5-sonnet-20241022", 'claude-3-5-1sonnet-20241022']:
            return ModelName.CLAUDE_3_5_SONNET
        elif model_name in ["claude-3-5-haiku", "claude-3-5-haiku-20241022"]:
            return ModelName.CLAUDE_3_5_HAIKU
        elif model_name in ["claude-3-opus", "claude-3-opus-20240229"]:
            return ModelName.CLAUDE_3_OPUS
        elif model_name in ["gpt-4o", "gpt-4o-2024-08-06"]:
            return ModelName.GPT_4O
        else:
            return ModelName.ERROR


def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def parse_rank_string_to_code(rank_str: str) -> str:
    """
    将 rank_str 转换为 code
    rank_str 存在的情况
        1. 1. Answer 1  \n2. Answer 5  \n3. Answer 6  \n4. Answer 2  \n5. Answer 3  \n6. Answer 4"
        2. 1.Answer 5 2.Answer 3 3.Answer 4 4.Answer 1 5.Answer 2 6.Answer 6'
        3. 1. Solution 1  \n2. Solution 5  \n3. Solution 6  \n4. Solution 2  \n5. Solution 3  \n6. Solution 4"
        4. 1.Solution - 5 2.Solution - 3 3.Solution - 4 4.Solution - 1 5.Solution - 2 6.Solution - 6
    得到 code 为 "156234"
    """
    # 处理单行情况，确保每个答案项之间有空格
    # 处理单行情况，确保每个答案项之间有空格
    if '\n' not in rank_str:
        rank_str = re.sub(r'(\d\.)\s*(Answer|Solution)', r'\1 \2', rank_str)

    # 分割字符串为行
    lines = []
    for line in rank_str.split('\n'):
        # 处理单行多个答案/solution的情况
        if (('Answer' in line or 'Solution' in line) and
            (line.count('Answer') + line.count('Solution') > 1)):
            # 使用正则表达式分割每个答案/solution项
            parts = re.split(r'(\d+\.\s*(?:Answer|Solution)\s*\d+)', line)
            # 过滤空字符串并清理
            lines.extend([p.strip() for p in parts if p.strip()])
        elif (('Answer' in line or 'Solution' in line) and
            (line.count('Answer') + line.count('Solution') == 1)):
            lines.append(line.strip())
        else:
            pass

    # 提取每个答案/solution的编号
    answer_order = []
    for line in lines:
        # 匹配 "数字. Answer 数字" 或 "数字. Solution 数字" 格式
        match = re.search(r'\d+\.\s*(?:Answer|Solution)\s*-?\s*(\d+)', line)
        if match:
            answer_order.append(match.group(1))

    answer_order = answer_order[:6]
    if not answer_order:
        return ""

    # # 将答案顺序转换为code
    # code = [''] * len(answer_order)
    # try:
    #     for position, answer_num in enumerate(answer_order, 1):
    #         idx = int(answer_num) - 1
    #         if 0 <= idx < len(code):
    #             code[idx] = str(position)
    # except (ValueError, IndexError) as e:
    #     print(f"Error processing answer order: {answer_order}, error: {e}")
    #     return ""

    return ''.join(answer_order)


def get_resps(*resps):
    resps = []
    for resp in resps:
        resps.append(parse_rank_string_to_code(resp))
    return resps


def main(datas: List[Tuple[str, List[Dict]]]):
    model_names, data_lists = zip(*datas) if datas else ([], [])
    if data_lists and not all(len(lst) == len(data_lists[0]) for lst in data_lists):
        raise ValueError("All data lists must have the same length")

    ranks = []
    for items in zip(*data_lists):
        questions = [item['doc']['question'] for item in items]
        if not all(q == questions[0] for q in questions[1:]):
            raise ValueError(f"Questions don't match for items: {questions}")
        responses = [item["filtered_resps"][0] for item in items]
        resps = [parse_rank_string_to_code(resp) for resp in responses]

        entry = {
            "doc_id": items[0]["doc_id"],
            "question": items[0]["doc"]["question"],
            "target": items[0]["doc"]["target"],
            "resps": resps[:len(datas)],
            "models": list(model_names)
        }
        if 'choices' in items[0]['doc']:
            entry['choices'] = items[0]['doc']['choices']
        ranks.append(entry)

    return ranks

if __name__ == "__main__":
    ROOT_PATH = 'lm-evaluation-harness/output/'
    benchmarks = os.listdir(ROOT_PATH)
    # benchmarks = ['gsm8k_cot']
    benchmarks = [os.path.join(ROOT_PATH, d, 'rank') for d in benchmarks if os.path.isdir(os.path.join(ROOT_PATH, d))]
    for benchmark in benchmarks:
        number_model_corresponse_path = os.path.join(benchmark, 'input', '对应关系.json')
        number_model_corresponse = read_json_file(number_model_corresponse_path)
        dirs = os.listdir(benchmark)
        model_dict = {}
        number_models = []
        for dir in dirs:
            model_name = ModelName.get_model_name(dir).value
            if model_name == 'error':
                continue
            number_model = number_model_corresponse.get(model_name)
            number_models.append(number_model)
        assert all(x == number_models[0] for x in number_models)
        number_model = number_models[0]
        number_model = sorted(number_model.items(), key=lambda x: x[0])

        model_dict_sorted = []
        for num, model_name in number_model:
            for dir in dirs:
                dir_name = ModelName.get_model_name(dir).value
                if dir_name == model_name:
                    model_dict_sorted.append(
                        (model_name, os.path.join(benchmark, dir))
                    )
                    break
                else:
                    continue

        print(model_dict_sorted)
        datas = []
        for model_name, model_path in model_dict_sorted:
            # Find the _rank.jsonl file in the model directory
            rank_files = [f for f in os.listdir(model_path) if f.endswith('_rank.jsonl')]
            if not rank_files:
                print(f"Warning: No _rank.jsonl file found in {model_path}")
                continue
            raw_rank_path = os.path.join(model_path, rank_files[0])
            data = read_json_file(raw_rank_path)
            datas.append((model_name, data))
        ranks = main(datas)
        output_path = os.path.join(benchmark, 'input', 'ranks.json')
        save_json_file(output_path, ranks)
        print(f"Output saved to {output_path}")
