# OpenCompass 评估指南：ifeval 和 MBPP

## 目录
1. [ifeval 评估](#ifeval-评估)
2. [MBPP 评估](#mbpp-评估)
3. [配置示例](#配置示例)
4. [运行评估](#运行评估)
5. [结果解读](#结果解读)

## ifeval 评估

### 评估内容
ifeval（Instruction Following Evaluation）用于评估模型遵循复杂指令的能力。

### 评估指标
1. **精确匹配（Exact Match）**
   - 严格检查模型输出是否与参考答案完全一致
   - 适用于有明确答案的任务

2. ROUGE 分数
   - ROUGE-N：n-gram 重叠率
   - ROUGE-L：最长公共子序列
   - 适用于生成式任务的评估

3. BLEU 分数
   - 计算生成文本与参考文本的 n-gram 重叠
   - 常用于机器翻译和文本生成任务

4. 自定义规则评分
   - 根据具体需求设计评分规则
   - 可以结合多个指标进行综合评分

## MBPP 评估

### 评估内容
MBPP（Mostly Basic Python Programming）用于评估模型编写正确 Python 代码的能力。

### 评估指标
1. **Pass@k**
   - 在前 k 个生成结果中，至少有一个通过所有测试用例的概率
   - 常用 k=1, 10, 100

2. **测试用例通过率**
   - 通过测试用例的比例
   - 可以细分为：
     - 完全通过率
     - 部分通过率

3. **代码质量评估**
   - 代码风格检查
   - 代码复杂度分析
   - 潜在 bug 检测

## 配置示例

### ifeval 配置示例
```yaml
datasets:
  - name: ifeval
    path: path/to/ifeval
    type: instruction_following
    metrics:
      - exact_match
      - rouge
      - bleu
```

### MBPP 配置示例
```yaml
datasets:
  - name: mbpp
    path: path/to/mbpp
    type: code_generation
    metrics:
      - pass_at_k: [1, 10, 100]
      - test_case_pass_rate
      - code_quality
```

## 运行评估

### 命令行示例
```bash
# 运行 ifeval 评估
python tools/run.py configs/eval_ifeval.py --model your_model

# 运行 MBPP 评估
python tools/run.py configs/eval_mbpp.py --model your_model
```

### 参数说明
- `--model`: 指定要评估的模型
- `--workers`: 工作进程数
- `--debug`: 调试模式
- `--out-dir`: 结果输出目录

## 结果解读

### ifeval 结果
```json
{
  "exact_match": 0.85,
  "rouge-1": 0.92,
  "rouge-2": 0.87,
  "rouge-l": 0.89,
  "bleu-4": 0.82
}
```

### MBPP 结果
```json
{
  "pass@1": 0.75,
  "pass@10": 0.92,
  "pass@100": 0.98,
  "test_case_pass_rate": 0.89,
  "code_quality": {
    "pylint_score": 9.2,
    "cyclomatic_complexity": 3.1
  }
}
```

## 注意事项

1. 确保测试环境配置正确
2. 检查数据集路径是否正确
3. 对于大型模型，注意显存使用情况
4. 建议在评估前进行小规模测试
5. 保存详细的评估日志以便后续分析
