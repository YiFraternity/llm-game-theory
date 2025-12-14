from scipy.stats import kendalltau

# 给定的 6 个排名
rankings = [
    [4,6,3,2,1,5],
    [3,4,1,6,5,2],
    [4,3,6,2,1,5],
    [4,6,3,2,1,5],
    [4,3,2,6,5,1],
    [4,3,2,6,1,5]
]

# 检查的综合排名
test_rank_1 = [1,4,6,5,3,2]  # 需要验证的排名
test_rank_2 = [4,5,2,3,1,6]  # 假设最优解

# 计算综合 Kendall Tau 和
def compute_total_kendall(rank, rankings):
    total_kendall = 0
    for r in rankings:
        print(r)
        print(kendalltau(rank, r))
        tau, _ = kendalltau(rank, r)
        total_kendall += tau
    return total_kendall

# 计算两个排名的 Kendall Tau 和
kendall_sum_1 = compute_total_kendall(test_rank_1, rankings)
kendall_sum_2 = compute_total_kendall(test_rank_2, rankings)

print(f"Kendall Tau 和 (1, 6, 4, 2, 5, 3): {kendall_sum_1}")
print(f"Kendall Tau 和 (2, 6, 3, 4, 5, 1): {kendall_sum_2}")
