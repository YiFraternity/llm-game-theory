import json
import os
from collections import defaultdict
import re

def merge_json_files(folder_path):
    # 使用字典存储所有文档，键为(question, tuple(choices))，值为resps列表
    documents_by_question = defaultdict(list)

    # 遍历folder_path下的所有子文件夹
    for subdir in next(os.walk(folder_path))[1]:
        subdir_path = os.path.join(folder_path, subdir)
        # 遍历子文件夹中的文件
        for filename in os.listdir(subdir_path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(subdir_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                    for doc in documents:
                        question_key = (doc['doc']['question'], tuple(doc['doc']['choices']))
                        for resp in doc['resps']:
                            # 假设resp是一个字符串列表，将每个字符串作为新的列表项添加
                            #resp[0] = re.sub(r'\n.*', '', resp[0], flags=re.DOTALL)
                            documents_by_question[question_key].extend(resp)
    # 准备所有文档的列表
    all_documents = []

    # 为每个合并后的文档分配新的doc_id，并添加到列表中
    for new_doc_id, ((question, choices), resps) in enumerate(documents_by_question.items()):
        new_doc = {
            'doc_id': new_doc_id,
            'question': question,
            'choices': list(choices),
            'resps': resps
        }
        all_documents.append(new_doc)

    # 创建输出目录
    output_folder_path = os.path.join(folder_path, 'input')
    os.makedirs(output_folder_path, exist_ok=True)

    # 将所有文档保存为一个JSON文件
    output_file_path = os.path.join(output_folder_path, "merged_documents.json")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=4)

# 使用示例
folder_path = '/Users/msy/Desktop/lm-evaluation-harness/lm_eval/output/mmlu_flan_cot_fewshot'  # 设置为包含子文件夹的文件夹路径
merge_json_files(folder_path)




