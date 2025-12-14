#!/usr/bin/env python3
"""
perturb_ranks.py

按比例随机选择评估问题，并对每个选定的问题随机选择一个模型，将该模型在每个排名字符串中移动一定位置以实现扰动。

假设每个条目具有字段:
 - "models": 列表，长度为 N
 - "resps": 列表，每个元素是一个长度为 N 的字符串，包含从 '1' 到 str(N) 的排列，表示模型的排名顺序

用法示例:
 python3 Q5_perturb_ranks.py \
     --input outputs/gsm8k/input/ranks.json \
     --output outputs/gsm8k/input/ranks_perturbed.json \
     --q_frac 0.1 --seed 42

参数说明:
 - q_frac: 在 [0,1] 内，选取的问题比例
 - seed: 随机种子，便于复现
 - dry_run: 只打印会被扰动的条目摘要但不写输出
 - sweep-q: 如果设置，则对 q_frac 从 0.0 到 1.0（步长 0.1）进行扫描，生成每个 q 的输出文件

输出: 将写入 --output 指定的文件（JSON），并在每个被扰动的条目中加入一个 metadata 字段 `__perturbation` 描述所做的修改。
"""

import argparse
import json
import random
from typing import List, Dict, Any, Optional


def clamp(x: int, a: int, b: int) -> int:
    return max(a, min(b, x))


def perturb_resp_string(resp: str) -> str:
    """
    生成一个新的随机排名（排列）作为扰动结果。
    现在不再仅移动单个数字，而是返回一个随机排列（由 '1'..str(n) 组成的字符串），
    尽量与原始 resp 不相同。
    """
    # 如果 resp 非字符串或为空，返回原始值
    if not isinstance(resp, str) or len(resp) == 0:
        return resp

    n = len(resp)
    digits = [str(i + 1) for i in range(n)]
    max_attempts = 10
    for _ in range(max_attempts):
        perm = ''.join(random.sample(digits, n))
        if perm != resp:
            return perm
    return resp


def perturb_dataset(data: List[Dict[str, Any]], q_frac: float, seed: Optional[int] = None, verbose: bool = False) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)

    n_docs = len(data)
    if n_docs == 0:
        return data

    n_select = max(1, int(round(n_docs * q_frac))) if q_frac > 0 else 0
    doc_indices = list(range(n_docs))
    selected = set(random.sample(doc_indices, n_select)) if n_select > 0 else set()

    out = []
    for i, entry in enumerate(data):
        entry_copy = dict(entry)  # shallow copy
        if i in selected:
            models = entry.get('models') or []
            resps = entry.get('resps') or []
            n_models = len(models)
            if n_models == 0 or any(not isinstance(r, str) for r in resps):
                # 无效格式，记录为已选中但跳过扰动（便于追踪 q_frac=1 情况）
                if verbose:
                    print(f"skip doc {i}: invalid models/resps format")
                entry_copy['__perturbation'] = {
                    'selected': True,
                    'skipped': True,
                    'reason': 'invalid models/resps format'
                }
                out.append(entry_copy)
                continue

            # 随机选择一个模型来扰动
            model_idx = random.randrange(n_models)

            # 仅对该题的一条响应进行扰动（随机选择一行），其它行保持不变
            resp_count = len(resps)
            new_resps = list(resps)  # shallow copy
            resp_idx = random.randrange(resp_count) if resp_count > 0 else None
            if resp_idx is not None:
                orig = resps[resp_idx]
                new_resp = perturb_resp_string(orig)
                new_resps[resp_idx] = new_resp

            entry_copy['resps'] = new_resps
            entry_copy['__perturbation'] = {
                'model_index': model_idx,
                'model_name': models[model_idx] if model_idx < len(models) else None,
                'selected': True,
                'resp_index_changed': resp_idx,
                'num_resps_changed': sum(1 for a, b in zip(resps, new_resps) if a != b)
            }
            if verbose:
                print(f"perturbed doc {i}: model {model_idx} ('{models[model_idx]}'), changed {entry_copy['__perturbation']['num_resps_changed']} / {len(resps)}")

        else:
            entry_copy['__perturbation'] = {'selected': False}

        out.append(entry_copy)

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='input ranks json file')
    p.add_argument('--output', '-o', required=True, help='output file path')
    p.add_argument('--q_frac', type=float, default=0.1, help='fraction of questions to perturb (0..1)')
    p.add_argument('--sweep-q', action='store_true', help='if set, run q_frac from 0.0 to 1.0 with step 0.1 and output one file per q')
    p.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    p.add_argument('--dry_run', action='store_true', help="don't write output, just print summary")
    p.add_argument('--verbose', action='store_true')

    args = p.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 支持对 q_frac 扫描（0.0..1.0）或单次运行
    if args.sweep_q:
        out_base = args.output
        if out_base.lower().endswith('.json'):
            base = out_base[:-5]
            ext = '.json'
        else:
            base = out_base
            ext = '.json'

        step = 0.1
        val = 0.0
        while val <= 1.0 + 1e-9:
            q = round(val, 1)
            pert = perturb_dataset(data, q_frac=q, seed=args.seed, verbose=args.verbose)
            n_total = len(pert)
            n_perturbed = sum(1 for e in pert if e.get('__perturbation', {}).get('selected'))
            n_changes = sum(e.get('__perturbation', {}).get('num_resps_changed', 0) for e in pert)
            print(f"q_frac={q:.1f}: docs: {n_total}, selected for perturbation: {n_perturbed}, total resp changes: {n_changes}")
            out_path = f"{base}_q{q:.1f}{ext}"
            if not args.dry_run:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(pert, f, ensure_ascii=False, indent=2)
                print(f"wrote perturbed output to {out_path}")
            val = round(val + step, 10)
    else:
        perturbed = perturb_dataset(data, q_frac=args.q_frac, seed=args.seed, verbose=args.verbose)
        # summary
        n_total = len(data)
        n_perturbed = sum(1 for e in perturbed if e.get('__perturbation', {}).get('selected'))
        n_changes = sum(e.get('__perturbation', {}).get('num_resps_changed', 0) for e in perturbed)
        print(f"docs: {n_total}, selected for perturbation: {n_perturbed}, total resp changes: {n_changes}")

        if not args.dry_run:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(perturbed, f, ensure_ascii=False, indent=2)
            print(f"wrote perturbed output to {args.output}")


if __name__ == '__main__':
    main()
