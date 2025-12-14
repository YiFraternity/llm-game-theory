models="gpt-4o-2024-05-13 gpt-4o-2024-08-06 gpt-4o-2024-11-20 claude-3-5-sonnet-20241022 claude-3-opus-20240229 claude-3-5-haiku-20241022"
benchmarks="overall"

for model in $models; do
    for benchmark in $benchmarks; do
        python run_inference.py \
            --config configs/Q4_"$benchmark"_ans.yaml \
            --model $model \
            --max_tokens 512 \
            --temperature 0.7 \
            --system_prompt "You are a helpful AI assistant." \
            --output_dir ./outputs/Q4/$benchmark/$model
    done
done

python merge_responses.py \
    --dataset-dir ./outputs/Q4/$benchmark \
    --output-file ./outputs/Q4/$benchmark/merged_results.json

models="gpt-4o-2024-05-13 gpt-4o-2024-08-06 gpt-4o-2024-11-20 claude-3-5-sonnet-20241022 claude-3-opus-20240229 claude-3-5-haiku-20241022"
benchmarks="overall"

for benchmark in $benchmarks; do
    for model in $models; do
        python run_inference.py \
            --config configs/Q4_"$benchmark"_ranks.yaml \
            --model $model \
            --max_tokens 128 \
            --temperature 0.7 \
            --system_prompt "You are a helpful AI assistant." \
            --output_dir ./outputs/Q4/$benchmark/ranks/$model
        echo "Finished $model for $benchmark"
    done
done

