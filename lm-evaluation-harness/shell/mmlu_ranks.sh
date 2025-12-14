export OPENAI_API_KEY=sk-FsJDWpqYgKxMMlDKoP5j4wCGN9OJyyLoSq1zki22YSiulXCO
OPENAI_API_BASE=https://api.chatanywhere.tech
cd /home/yhliu/game-theory/lm-evaluation-harness

models="gpt-4o-2024-05-13 gpt-4o-2024-08-06 gpt-4o-2024-11-20 claude-3-5-sonnet-20241022 claude-3-opus-20240229"
benchmark="mmlu"
for model in $models
do
    lm_eval --model openai-chat-completions \
        --model_args model=$model,base_url=$OPENAI_API_BASE,max_tokens=1024 \
        --tasks rank \
        --output output/$benchmark/rank/$model \
        --log_samples
done