export OPENAI_API_KEY=sk-FsJDWpqYgKxMMlDKoP5j4wCGN9OJyyLoSq1zki22YSiulXCO
OPENAI_API_BASE=https://api.chatanywhere.tech
cd /home/yhliu/game-theory/lm-evaluation-harness

models="claude-3-5-haiku-20241022"
benchmark="gpqa"
for model in $models
do
    lm_eval --model openai-chat-completions \
        --model_args model=$model,base_url=$OPENAI_API_BASE,max_tokens=1024 \
        --tasks rank \
        --output output/$benchmark/rank/$model \
        --log_samples
done