from typing import List, Dict, Any, Optional, Union, Tuple
import re
import json
import os
import copy
import math
import random
import itertools
import collections
from itertools import permutations
import yaml
import numpy as np
from scipy.stats import spearmanr, kendalltau
import pandas as pd


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the ranks data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the ranks data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_yaml_data(file_path: str) -> Dict[str, Any]:
    """Load the ranks data from YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_json_data(data: Union[Dict[str, Any], List[Dict[str, Any]]], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_jsonl_data(data: Union[Dict[str, Any], List[Dict[str, Any]]], file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


MMLU_MODEL_ID_NAME = {
    '1': 'chatgpt-4o-latest-20241120',
    '2': 'gpt-4o-2024-05-13',
    '3': 'claude-3-5-sonnet-20241022',
    '4': 'claude-3-5-haiku-20241022',
    '5': 'claude-3-opus-20240229',
    '6': 'gpt-4o-2024-08-06',
}

MBPP_MODEL_ID_NAME = {
    "1": "gpt-4o-2024-11-20",
    "2": "claude-3-5-sonnet-20241022",
    "3": "gpt-4o-2024-05-13",
    "4": "gpt-4o-2024-08-06",
    "5": "claude-3-opus-20240229",
    "6": "claude-3-5-haiku-20241022",
}

IFEVAL_MODEL_ID_NAME = {
    '1': 'gpt-4o-2024-11-20',
    '2': 'claude-3-5-sonnet-20241022',
    '3': 'gpt-4o-2024-05-13',
    '4': 'gpt-4o-2024-08-06',
    '5': 'claude-3-opus-20240229',
    '6': 'claude-3-5-haiku-20241022',
}

WRITING_BENCH_MODEL_ID_NAME = {
    "1": "gpt-4o-2024-11-20",
    "2": "claude-3-5-sonnet-20241022",
    "3": "gpt-4o-2024-05-13",
    "4": "gpt-4o-2024-08-06",
    "5": "claude-3-opus-20240229",
    "6": "claude-3-5-haiku-20241022",
}

CEVAL_MODEL_ID_NAME = {
    "1": "gpt-4o-2024-11-20",
    "2": "gpt-4o-2024-05-13",
    "3": "claude-3-5-sonnet-20241022",
    "4": "gpt-4o-2024-08-06",
    "5": "claude-3-opus-20240229",
    "6": "claude-3-5-haiku-20241022",
}

GPQA_MODEL_ID_NAME = {
    "1": "chatgpt-4o-latest-20241120",
    "2": "gpt-4o-2024-05-13",
    "3": "claude-3-5-sonnet-20241022",
    "4": "claude-3-5-haiku-20241022",
    "5": "claude-3-opus-20240229",
    "6": "gpt-4o-2024-08-06"
}

GSM8K_MODEL_ID_NAME = {
    "1": "gpt-4o-2024-11-20",
    "2": "claude-3-5-sonnet-20241022",
    "3": "gpt-4o-2024-05-13",
    "4": "gpt-4o-2024-08-06",
    "5": "claude-3-5-haiku-20241022",
    "6": "claude-3-opus-20240229"
}

OVERALL_RANKING = {
    'chatgpt-4o-latest-20241120': 1,
    'gpt-4o-2024-05-13': 2,
    'claude-3-5-sonnet-20241022': 3,
    'gpt-4o-2024-08-06': 4,
    'claude-3-opus-20240229': 5,
    'claude-3-5-haiku-20241022': 6,
}

GEN_OVERALL_RANKING = {
    'gpt-4o-2024-11-20': 1,
    'gpt-4o-2024-05-13': 2,
    'claude-3-5-sonnet-20241022': 3,
    'gpt-4o-2024-08-06': 4,
    'claude-3-opus-20240229': 5,
    'claude-3-5-haiku-20241022': 6,
}

CHATBOT_ARENA_RANK = {
    'chatgpt-4o-latest-20241120': 1,
    'gpt-4o-2024-05-13': 2,
    'claude-3-5-sonnet-20241022': 3,
    'gpt-4o-2024-08-06': 4,
    'claude-3-5-haiku-20241022': 5,
    'claude-3-opus-20240229': 6,
}

CHINESE_RANKING = {
    "gpt-4o-2024-11-20": 1,
    "gpt-4o-2024-05-13": 2,
    "claude-3-5-sonnet-20241022": 3,
    "gpt-4o-2024-08-06": 4,
    "claude-3-opus-20240229": 5,
    "claude-3-5-haiku-20241022": 6,
}

INSTRUCTION_RANKING = {
    "gpt-4o-2024-11-20": 1,
    "claude-3-5-sonnet-20241022": 2,
    "gpt-4o-2024-05-13": 3,
    "gpt-4o-2024-08-06": 4,
    "claude-3-opus-20240229": 5,
    "claude-3-5-haiku-20241022": 6,
}

CREATING_WRITING = {
    "gpt-4o-2024-11-20": 1,
    "claude-3-5-sonnet-20241022": 2,
    "gpt-4o-2024-05-13": 3,
    "gpt-4o-2024-08-06": 4,
    "claude-3-opus-20240229": 5,
    "claude-3-5-haiku-20241022": 6,
}

CODEING = {
    "gpt-4o-2024-11-20": 1,
    "claude-3-5-sonnet-20241022": 2,
    "gpt-4o-2024-05-13": 3,
    "gpt-4o-2024-08-06": 4,
    "claude-3-5-haiku-20241022": 5,
    "claude-3-opus-20240229": 6,
}

MATHEMATIC = {
    "gpt-4o-2024-11-20": 1,
    "claude-3-5-sonnet-20241022": 2,
    "gpt-4o-2024-05-13": 3,
    "gpt-4o-2024-08-06": 4,
    "claude-3-opus-20240229": 5,
    "claude-3-5-haiku-20241022": 6,
}

# 计算Borda排名
def borda_rank(rankings: List[List[int | None]]) -> List[int]:
    """
    计算Borda排名，
    Rankings: List[List[int | None]]
        每行表示一个选民的投票，列表示候选人的排名。
        例如：rankings[i][j]表示第i个选民对第j个候选人的排名。
    基本思想：
        1. 每个投票者对所有候选项进行排序
        2. 排名第一的候选项得分为n-1分，第二名为n-2分，依此类推，最后一名得0分
        3. 将所有投票者的得分相加，得到每个候选项的总分
        4. 按总分从高到低排序，得到指定候选者的最终排名
    """
    n_candidates = len(rankings[0])  # 候选人数
    borda_scores = [0] * n_candidates

    for voter_rankings in rankings:  # 每个投票者的排名
        for candidate_idx in range(n_candidates):
            rank = voter_rankings[candidate_idx]
            if rank is not None:
                borda_scores[candidate_idx] += (n_candidates - rank)

    ranked_indices = sorted(range(n_candidates), key=lambda i: -borda_scores[i])
    ranks = [0] * n_candidates
    for rank, idx in enumerate(ranked_indices, 1):
        ranks[idx] = rank
    return ranks

def average_rank(rankings: List[List[int | None]]) -> List:
    """
    计算平均排名，平均分数越高，排名越靠前
    rankings: List[List[int | None]]
        每行表示一个选民的投票，列表示候选人的排名。
    1. 计算每个候选人的平均排名
    2. 按平均排名从低到高排序
    """
    return np.mean(rankings, axis=0).tolist()
    # n = len(rankings[0])
    # average_scores = {}
    # for ranking in rankings:
    #     for i in range(n):
    #         if ranking[i] is not None:
    #             if ranking[i] not in average_scores:
    #                 average_scores[ranking[i]] = 0
    #             average_scores[ranking[i]] += i+1
    # average_scores = {k: v / len(rankings) for k, v in average_scores.items()}

    # return [i for i, _ in sorted(average_scores.items(), key=lambda x: x[1], reverse=False)]

def copeland_rank(rankings: List[List[int | None]]) -> List[int]:
    """
    使用Copeland法对候选人进行排名，输入为一个候选人之间的排名矩阵（6x6），返回排名。
    Rankings: List[List[int | None]]
        每行表示一个选民的投票，列表示候选人的排名。
        例如：rankings[i][j]表示第i个选民对第j个候选人的排名。
    Copeland法：
        1. 对每对候选人进行两两比较
        2. 如果候选人A在多数选票中排在候选人B前面，则A得1分，B得-1分
        3. 如果平局，则都得0分
        4. 最终根据总分进行排名
    """
    num_candidates = len(rankings[0]) if rankings else 0
    if num_candidates == 0:
        return []
    scores = {i: 0 for i in range(num_candidates)}
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            i_wins = 0
            j_wins = 0

            for vote in rankings:
                if vote[i] is None or vote[j] is None:
                    continue
                if vote[i] < vote[j]:  # i排在j前面
                    i_wins += 1
                elif vote[j] < vote[i]:  # j排在i前面
                    j_wins += 1

            if i_wins > j_wins:
                scores[i] += 1
                scores[j] -= 1
            elif j_wins > i_wins:
                scores[i] -= 1
                scores[j] += 1

    sorted_candidates = sorted(scores.keys(), key=lambda x: (-scores[x], x))
    rank = [0] * num_candidates
    for position, candidate in enumerate(sorted_candidates, 1):
        rank[candidate] = position

    return rank

def condorcet_method(rankings: List[List[int | None]]) -> List[int]:
    """
    使用孔多塞法对候选人进行排名，输入为一个候选人之间的排名矩阵（6x6），返回`1-based`排名。
    其实基本就是copeland法。
    Rankings: List[List[int | None]]
        每行表示一个选民的投票，列表示候选人的排名。
        例如：rankings[i][j]表示第i个选民对第j个候选人的排名。
    """
    num_candidates = len(rankings[0])
    victories = [[0 for _ in range(num_candidates)] for _ in range(num_candidates)]

    # 构造胜负矩阵 victories[i][j]：表示 i 胜过 j 的选民数量
    for vote in rankings:      # 当前选民的投票
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                i_rank = vote[i]
                j_rank = vote[j]
                if i_rank < j_rank:
                    victories[i][j] += 1
                elif i_rank > j_rank:
                    victories[j][i] += 1
    # 判断是否存在 Condorcet Winner
    condorcet_winner = None
    for i in range(num_candidates):
        is_winner = True
        for j in range(num_candidates):
            if i == j:
                continue
            if victories[i][j] <= victories[j][i]:  # i 不能赢 j
                is_winner = False
                break
        if is_winner:
            condorcet_winner = i
            break

    # Copeland 得分计算
    copeland_scores = [0] * num_candidates
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i == j:
                continue
            if victories[i][j] > victories[j][i]:
                copeland_scores[i] += 1
            elif victories[i][j] < victories[j][i]:
                copeland_scores[i] -= 1
            # 平局不加分

    # 排名：得分高的排前，索引小作为平局破局
    sorted_candidates = sorted(
        range(num_candidates),
        key=lambda x: (-copeland_scores[x], x)
    )

    # 如果有 Condorcet Winner，把他放第一
    if condorcet_winner is not None:
        sorted_candidates.remove(condorcet_winner)
        sorted_candidates = [condorcet_winner] + sorted_candidates

    # 转换为 1-based 排名
    final_rank = [0] * num_candidates
    for position, candidate in enumerate(sorted_candidates, 1):
        final_rank[candidate] = position

    return final_rank


def is_condorcet_winner(pref_profile, cand, n_cand):
    """判断 cand 是否在当前偏好档下是 Condorcet Winner"""
    for opp in range(n_cand):
        if opp == cand:
            continue
        wins = sum(vote.index(cand) < vote.index(opp) for vote in pref_profile)
        losses = len(pref_profile) - wins
        if wins <= losses:
            return False
    return True

def dodgson_score(profile: List[List[int]], cand: int, n_cand: int) -> int:
    """
    通过计数每个候选人和其他候选人的胜负关系来估算 Dodgson 分数
    """
    # 计算每个候选人相对于其他候选人的胜负关系
    score = 0
    for i in range(n_cand):
        if i == cand:
            continue
        wins = sum(vote.index(cand) < vote.index(i) for vote in profile)  # cand 胜 i 的次数
        losses = len(profile) - wins  # cand 输 i 的次数

        if wins <= losses:
            score += (losses - wins)  # 每次失败的候选人需要更多的交换来翻转结果
    return score


def dodgson_method(rankings: List[List[Optional[int]]]) -> List[int]:
    """
    使用 Dodgson 法对候选人进行排名，返回每个候选人的最终排名。
    """
    profile = []
    for vote in rankings:
        pair = [(rank, idx) for idx, rank in enumerate(vote) if rank is not None]
        pair.sort()  # 对每个选民的投票进行排序
        profile.append([idx for _, idx in pair])  # 得到选民对候选人的偏好排序

    n_cand = len(rankings[0])  # 假设每行的长度相同，即候选人数量
    scores = [dodgson_score(profile, c, n_cand) for c in range(n_cand)]  # 计算每个候选人的 Dodgson 分数
    order = sorted(range(n_cand), key=lambda x: (scores[x], x))  # 根据 Dodgson 分数对候选人进行排序

    # 反向映射：候选人 index -> ranking
    ranks = [0] * n_cand
    for place, cand in enumerate(order):
        ranks[cand] = place + 1  # 为每个候选人分配排名

    return ranks  # 返回每个候选人的最终排名（1-based）


def kendall_tau(rankings: List[List[int | None]]) -> List[int]:
    """
    使用Kendall Tau法对候选人进行排名，输入为一个候选人之间的排名矩阵（6x6），返回每个候选人的排名。
    Kendall Tau法：
    1. 枚举所有可能的排名排列
    2. 计算每个排列与所有输入排名的Kendall Tau相关性之和
    3. 选择使总相关性最大的排列作为最优排名

    Args:
        rankings: 一个候选人之间的排名矩阵（6x6），
            每行表示一个选民的投票，列表示候选人的排名。
    Returns:
        每个候选人的最终排名（1-based）。
    """
    rankings = np.array(rankings, dtype=float)
    valid_rows = ~np.isnan(rankings).any(axis=1)
    rankings = rankings[valid_rows].astype(int)

    if len(rankings) == 0:
        raise ValueError("有效数据为空")

    n_candidates = rankings.shape[1]

    candidates = list(range(n_candidates))
    best_score = -float('inf')
    best_ranking = None

    # 枚举所有候选人排列并计算Kendall Tau相关性
    for perm in permutations(candidates):
        current_ranking = [x + 1 for x in perm]
        total_tau = 0

        for rank in rankings:
            tau, _ = kendalltau(current_ranking, rank)
            total_tau += tau

        if total_tau > best_score:
            best_score = total_tau
            best_ranking = current_ranking

    return best_ranking


def kemeny_young_method(rankings: List[List[int | None]]) -> List[int]:
    """
    使用Kemeny-Young法对候选人进行排名，输入为一个候选人之间的排名矩阵（6x6），返回排名。
    Kemeny距离：
        Kemeny距离用于衡量两个全序排名之间的差异，其计算核心是统计对应位置上候选人不同的数量。

    Kemeny距离计算示例：
        候选人A：A > B > C（表示为 [A, B, C]）
        候选人B：B > A > C（表示为 [B, A, C]）
        候选人C：A > C > B（表示为 [A, C, B]）

        1. 计算排名1与排名2的Kemeny距离
           - 位置1：A vs B → 不同，+1
           - 位置2：B vs A → 不同，+1
           - 位置3：C vs C → 相同，+0
           总距离：1 + 1 = 2

        2. 计算排名1与排名3的Kemeny距离
           - 位置1：A vs A → 相同，+0
           - 位置2：B vs C → 不同，+1
           - 位置3：C vs B → 不同，+1
           总距离：0 + 1 + 1 = 2

        3. 计算排名2与排名3的Kemeny距离
           - 位置1：B vs A → 不同，+1
           - 位置2：A vs C → 不同，+1
           - 位置3：C vs B → 不同，+1
           总距离：1 + 1 + 1 = 3

    Kemeny-Young法：
        1. 构建偏好矩阵，统计每对候选人的偏好关系
        2. 枚举所有可能的排名，计算每个排名与所有投票的Kemeny距离
        3. 选择使总Kemeny距离最小的排名作为最优解

    Args:
        rankings: 一个候选人之间的排名矩阵（6x6），每行表示一个选民的投票，列表示候选人的顺序。
    Return：
        按Kemeny-Young法排序后的候选人排名列表（1-based，1表示最高）
    """
    rankings = np.array(rankings, dtype=float)
    valid_rows = ~np.isnan(rankings).any(axis=1)
    rankings = rankings[valid_rows].astype(int) - 1  # 转换为0-based索引

    if len(rankings) == 0:
        raise ValueError("有效数据为空")

    num_candidates = rankings.shape[1]

    preferences = np.zeros((num_candidates, num_candidates))

    for vote in rankings:
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                if vote[i] < vote[j]:
                    preferences[i][j] += 1
                elif vote[i] > vote[j]:
                    preferences[j][i] += 1

    all_rankings = list(itertools.permutations(range(num_candidates)))
    min_distance = float('inf')
    best_ranking = None

    for ranking in all_rankings:
        distance = 0
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                if ranking.index(i) < ranking.index(j):
                    distance += preferences[j][i]
                else:
                    distance += preferences[i][j]
        if distance < min_distance:
            min_distance = distance
            best_ranking = ranking
    ranks = [0] * num_candidates
    for rank, candidate in enumerate(best_ranking, 1):
        ranks[candidate] = rank

    return ranks

def convert_to_rank_list(order: List[int], n_candidates: int) -> List[int]:
    """将候选人ID顺序转换为1-based排名列表。"""
    rank_list = [0] * n_candidates
    for rank, cid in enumerate(order, 1):
        rank_list[cid] = rank
    return rank_list


def irv_method(rankings: List[List[Optional[int]]]) -> List[int]:
    """
    使用IRV法对候选人进行完整排名（从第一名到最后一名）。

    Args:
        rankings: 每行表示一个选民的投票，列为候选人，值为排名（整数），None为未填。

    Returns:
        List[int]: 每个候选人的1-based排名，索引代表候选人ID。
    """
    valid_rankings_np = np.array(rankings, dtype=int)
    n_candidates = valid_rankings_np.shape[1]

    # 每张选票按偏好排序的候选人ID列表
    processed_ballots = []
    for ballot_ranks in valid_rankings_np:
        sorted_ids = [idx for idx, _ in sorted(enumerate(ballot_ranks), key=lambda x: x[1])]
        processed_ballots.append(sorted_ids)

    active_candidates = set(range(n_candidates))
    elimination_order = []  # 淘汰顺序（先淘汰的在前）

    while len(active_candidates) > 1:
        vote_counts = {c: 0 for c in active_candidates}
        total_valid_votes = 0

        for ballot in processed_ballots:
            for candidate_id in ballot:
                if candidate_id in active_candidates:
                    vote_counts[candidate_id] += 1
                    total_valid_votes += 1
                    break

        if total_valid_votes == 0:
            # 全部弃权的情况
            break

        # 找出得票最少的候选人
        min_votes = min(vote_counts.values())
        min_candidates = [c for c in active_candidates if vote_counts[c] == min_votes]
        to_eliminate = min(min_candidates)  # 打破平局：选择ID最小的淘汰
        elimination_order.append(to_eliminate)
        active_candidates.remove(to_eliminate)

    # 最后一个幸存者是第一名
    if len(active_candidates) == 1:
        final_order_ids = [next(iter(active_candidates))] + elimination_order[::-1]
    else:
        # 如果全部弃权，剩下的按ID排序
        final_order_ids = sorted(active_candidates) + elimination_order[::-1]

    # 转为1-based排名（索引是候选人ID，值是名次）
    rank_output = [0] * n_candidates
    for rank, candidate_id in enumerate(final_order_ids, 1):
        rank_output[candidate_id] = rank

    return rank_output


def spearmanr_ranking(rankings: List[List[int | None]]) -> List[int]:
    """
    使用Spearman相关系数对候选人进行聚合排名。
    Args:
        rankings: 一个候选人之间的排名矩阵（行是投票者，列是候选人），1-based排名。
    Returns:
        每个候选人的最终排名（索引0是候选人0的名次）
    """
    if not rankings:
        return []

    n_candidates = len(rankings[0])

    # 过滤掉包含None的行
    valid_rankings = [r for r in rankings if all(x is not None for x in r)]
    if not valid_rankings:
        return [0] * n_candidates

    valid_rankings = np.array(valid_rankings)

    max_spearman = -float('inf')
    best_rank = None

    for perm in permutations(range(n_candidates)):  # perm是候选人编号顺序
        # 转换成每个候选人的“排名”列表：index是候选人，值是他们的名次
        rank_of_perm = [0] * n_candidates
        for pos, cand in enumerate(perm):
            rank_of_perm[cand] = pos + 1  # 1-based

        total_rho = 0
        for vote in valid_rankings:
            rho, _ = spearmanr(rank_of_perm, vote)
            if not np.isnan(rho):
                total_rho += rho

        if total_rho > max_spearman:
            max_spearman = total_rho
            best_rank = rank_of_perm

    return best_rank

class Rerank:
    """
    A utility class for applying different ranking methods to a set of rankings.

    This class provides methods to compute various ranking aggregations including
    Kendall's tau, Borda count, average rank, and Spearman's rank correlation.

    Args:
        rankings: A list of rankings, where each ranking is a list of candidate scores or ranks.
                 Each sublist represents a single ranking of candidates.
    """

    def __init__(self, rankings: List[List[int | None]]):
        self.rankings = rankings

    def dodgeson(self):
        return dodgson_method(self.rankings)

    def kendall(self):
        return kendall_tau(self.rankings)

    def condorcet(self):
        return condorcet_method(self.rankings)

    def borda(self):
        return borda_rank(self.rankings)

    def copeland(self):
        return copeland_rank(self.rankings)

    def average(self):
        return average_rank(self.rankings)

    def spearman(self):
        return spearmanr_ranking(self.rankings)

    def kemeny_young(self):
        return kemeny_young_method(self.rankings)

    def irv(self):
        return irv_method(self.rankings)

    def rerank_method(self, method: str):
        """
        Rerank the rankings using the specified method.

        Args:
            method: The method to use for reranking. Must be one of 'kendall', 'borda', 'average', 'spearman', 'kemeny_young', 'irv', 'copeland', 'condorcet', 'dodgeson'.

        Returns:
            The reranked rankings.
        """
        assert method in ['kendall', 'borda', 'average', 'spearman', 'kemeny_young', 'irv', 'copeland', 'condorcet', 'dodgeson'], f"Invalid method: {method}"
        if method == 'kendall':
            return self.kendall()
        elif method == 'borda':
            return self.borda()
        elif method == 'copeland':
            return self.copeland()
        elif method == 'average':
            return self.average()
        elif method == 'spearman':
            return self.spearman()
        elif method == 'kemeny_young':
            return self.kemeny_young()
        elif method == 'irv':
            return self.irv()
        elif method == 'condorcet':
            return self.condorcet()
        elif method == 'dodgeson':
            return self.dodgeson()
        else:
            raise ValueError(f"Invalid method: {method}")


def count_text_components(text: str) -> int:
    """
    统计文本中的中文字符、英文单词和标点符号的数量

    Args:
        text: 需要统计的文本

    Returns:
        文本中的中文字符、英文单词和标点符号的数量
    """

    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    punctuation = re.findall(
        r'[。，、；：！？「」『』（）【】《》.,;:!?(){}\[\]"\'…]|\s-\s|--',
        text
    )
    return len(chinese_chars) + len(english_words) + len(punctuation)


def populate_template(template: str, variables: dict[str, Any]) -> str:
    from jinja2 import Template, StrictUndefined
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

def load_llm(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    gpu_num: int = 1,
    lora_model_name_or_path: Optional[str] = None,
    *,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Load a vLLM model and build default SamplingParams.

    Args:
        model_name_or_path: HF model name or local path.
        tokenizer_name_or_path: HF tokenizer name or local path. Defaults to model_name_or_path.
        gpu_num: Tensor parallel size.
        lora_model_name_or_path: If provided, enables LoRA. (Note: actual LoRA loading may require extra args.)
        llm_kwargs: Extra kwargs forwarded to vllm.LLM(...).
        sampling_kwargs: Extra kwargs forwarded to vllm.SamplingParams(...).

    Returns:
        (llm, sampling_params)
    """
    from vllm import LLM, SamplingParams
    if gpu_num < 1:
        raise ValueError(f"gpu_num must be >= 1, got {gpu_num}")

    llm_kwargs = dict(llm_kwargs or {})
    sampling_kwargs = dict(sampling_kwargs or {})

    llm_init_kwargs: Dict[str, Any] = {
        "model": model_name_or_path,
        "tokenizer": tokenizer_name_or_path or model_name_or_path,
        "tokenizer_mode": "slow",
        "tensor_parallel_size": gpu_num,
        "enable_lora": bool(lora_model_name_or_path),
        **llm_kwargs,
    }

    llm = LLM(**llm_init_kwargs)

    default_sampling: Dict[str, Any] = {
        "n": 1,
        "max_tokens": 1024,
        "top_p": 1.0,
        "temperature": 0.0,
        "top_k": 1,
    }

    sampling_params = SamplingParams(**{**default_sampling, **sampling_kwargs})

    return llm, sampling_params

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

if __name__ == '__main__':
    # 测试文本统计功能
    test_text = """
    这是一个测试文本，包含中文和English words。
    还有标点符号！比如：，。；、！？和英文标点,.;!?"'...
    """
    print("文本统计结果:", count_text_components(test_text))

    # 测试用例1：所有候选人都出现在每个投票中
    ranks1 = [
        [5, 1, 2, 4, 6, 3],
        [2, 1, 3, 4, 5, 6],
    ]
    print("最终排名:", dodgson_method(ranks1))


def get_rank_matrix(data: List[Dict[str, Any]], model_id_name: Dict[str, str], capability: Dict[str, int]) -> List[pd.DataFrame]:
    """
    针对每一道题，得到一个排名矩阵。返回所有题目的矩阵
    最终矩阵的列是capability中模型的排名。
    """
    if not data:
        return []
    models = data[0]['models']
    rank_matrix = []
    model_rankings = sorted(capability.items(), key=lambda x: x[1])
    model_rankings = [x[0] for x in model_rankings]
    for item in data:
        matrix = []
        cur_models = item['models']
        assert set(cur_models) == set(models)
        for resp in item['resps']:
            if len(resp) < 6:
                continue
            resp_id = resp[:6]
            resp_id = list(resp_id)
            resp_model_rank = {model_id_name[str(i+1)]: int(c) for i, c in enumerate(resp_id)}
            matrix.append(resp_model_rank)
        matrix_df = pd.DataFrame(matrix)
        assert set(matrix_df.columns.tolist()) == set(model_rankings)
        matrix_df = matrix_df.reindex(model_rankings, axis=1)
        rank_matrix.append(matrix_df)
    return rank_matrix