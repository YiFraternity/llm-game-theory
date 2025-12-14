# Ranking Methods Documentation

This document provides detailed information about the ranking methods implemented in `utils.py`. These methods are designed to aggregate multiple rankings into a single consensus ranking using different voting systems and statistical methods.

## Table of Contents

1. [Borda Count](#borda-count)
2. [Average Rank](#average-rank)
3. [Condorcet Method](#condorcet-method)
4. [Dodgson's Method](#dodgsons-method)
5. [Kendall Tau](#kendall-tau)
6. [Kemeny-Young Method](#kemeny-young-method)
7. [Instant-Runoff Voting (IRV)](#instant-runoff-voting-irv)
8. [Spearman's Rank Correlation](#spearmans-rank-correlation)

## Borda Count

### Function: `borda_rank(rankings: List[List[int | None]]) -> List[int]`

**Description:**
Implements the Borda count voting method, where each candidate receives points based on their position in each voter's ranking. The candidate with the highest total score wins.

**Parameters:**
- `rankings`: A list of rankings, where each ranking is a list of integers representing candidate positions (1-based).

**Returns:**
- A list of integers representing the final ranking (1-based, where 1 is the highest rank).

**Example:**
```python
rankings = [
    [1, 2, 3],
    [2, 1, 3],
    [3, 1, 2]
]
print(borda_rank(rankings))
# Output: [2, 1, 3]  # Candidate 2 wins, followed by 1, then 3
```

## Average Rank

### Function: `average_rank(rankings: List[List[int | None]]) -> List[int]`

**Description:**
Calculates the average position of each candidate across all rankings and returns the final ranking based on these averages.

**Parameters:**
- `rankings`: A list of rankings, where each ranking is a list of integers representing candidate positions.

**Returns:**
- A list of integers representing the final ranking based on average positions.

## Condorcet Method

### Function: `condorcet_method(rankings: List[List[int | None]]) -> List[int]`

**Description:**
Implements the Condorcet method, which finds a candidate who would win a head-to-head election against every other candidate.

**Parameters:**
- `rankings`: A list of rankings, where each ranking is a list of integers representing candidate positions.

**Returns:**
- A list of integers representing the final ranking based on pairwise comparisons.

## Dodgson's Method

### Function: `dodgson_method(rankings: List[List[int | None]]) -> List[int]`

**Description:**
Implements Dodgson's method, which finds the candidate closest to being a Condorcet winner by counting the minimum number of adjacent swaps needed to make each candidate a Condorcet winner.

**Parameters:**
- `rankings`: A list of rankings, where each ranking is a list of integers representing candidate positions.

**Returns:**
- A list of integers representing the final ranking based on Dodgson scores.

## Kendall Tau

### Function: `kendall_tau(rankings: List[List[int | None]]) -> List[int]`

**Description:**
Finds the ranking that has the highest total Kendall Tau correlation with all input rankings.

**Parameters:**
- `rankings`: A list of rankings, where each ranking is a list of integers representing candidate positions.

**Returns:**
- A list of integers representing the final ranking that maximizes Kendall Tau correlation.

## Kemeny-Young Method

### Function: `kemeny_young_method(rankings: List[List[int | None]]) -> List[int]`

**Description:**
Implements the Kemeny-Young method, which finds the ranking that minimizes the total number of pairwise disagreements with all input rankings.

**Parameters:**
- `rankings`: A list of rankings, where each ranking is a list of integers representing candidate positions.

**Returns:**
- A list of integers representing the final ranking that minimizes Kemeny distance.

## Instant-Runoff Voting (IRV)

### Function: `irv_method(rankings: List[List[int | None]]) -> List[int]`

**Description:**
Implements the Instant-Runoff Voting (IRV) method, which eliminates the candidate with the fewest first-preference votes in each round until one candidate has a majority.

**Parameters:**
- `rankings`: A list of rankings, where each ranking is a list of integers representing candidate positions.

**Returns:**
- A list of integers representing the final ranking based on elimination order.

## Spearman's Rank Correlation

### Function: `spearmanr_ranking(rankings: List[List[int | None]]) -> List[int]`

**Description:**
Finds the ranking that has the highest total Spearman's rank correlation with all input rankings.

**Parameters:**
- `rankings`: A list of rankings, where each ranking is a list of integers representing candidate positions.

**Returns:**
- A list of integers representing the final ranking that maximizes Spearman's rank correlation.

## Usage Example

```python
from utils import borda_rank, condorcet_method, irv_method

# Example rankings
rankings = [
    [1, 2, 3, 4],
    [2, 1, 3, 4],
    [3, 1, 2, 4],
    [4, 1, 2, 3]
]

# Get consensus ranking using different methods
print("Borda Count:", borda_rank(rankings))
print("Condorcet Method:", condorcet_method(rankings))
print("IRV Method:", irv_method(rankings))
```

## Notes

- All ranking methods expect 1-based rankings (1 is the highest rank).
- Missing or invalid rankings (None values) are automatically handled by most methods.
- Some methods (like Kemeny-Young) can be computationally expensive for large numbers of candidates due to the need to evaluate all possible permutations.
