```sh
lm_eval --model openai-chat-completions \
    --model_args model=chatgpt-4o-latest \
    --tasks gpqa_main_cot_n_shot \
    --output output/gpqa_main_cot_n_shot/chatgpt-4o-latest \
    --limit 50 \
    --write_out \
    --log_samples
```

```sh
lm_eval --model openai-chat-completions \
    --model_args model=chatgpt-4o-latest \
    --tasks rank \
    --output output/gpqa/rank/chatgpt-4o-latest \
    --log_samples
```

# Chatbot Arena
中文
    ```json
    {
        "GPT-4o-2024-05-13": 46,
        "GPT-4o-2024-08-06": 52,
        "Claude 3.5 Sonnet (20241022)": 48,
        "Claude 3.5 Haiku (20241022)": 63,
        "Claude 3.5 Opus (20240229)": 54,
    }
