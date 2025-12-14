export HF_ENDPOINT="https://hf-mirror.com"
export OPENAI_API_KEY=sk-FsJDWpqYgKxMMlDKoP5j4wCGN9OJyyLoSq1zki22YSiulXCO
OPENAI_API_BASE=https://api.chatanywhere.tech
# cd /home/yhliu/game-theory/lm-evaluation-harness

model=gpt-4o-2024-05-13
task=ceval-valid_chinese_language_and_literature
lm_eval --model openai-chat-completions \
    --model_args model=$model,base_url=$OPENAI_API_BASE,max_tokens=4096,temperature=0 \
    --tasks $task \
    --output output/$task/$model \
    --limit 50 \
    --write_out \
    --log_samples \
    --batch_size 1 \
    --apply_chat_template
