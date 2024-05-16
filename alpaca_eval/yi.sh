export RAY_memory_monitor_refresh_ms=0

test_path="data/test.json"
MODEL="../model/Yi-34B"
output_path="yi/yi.jsonl"
CUDA_VISIBLE_DEVICES=0 python vllm_inference.py \
        --data_file $test_path --output_file $output_path --base_model $MODEL \
        --tensor_parallel_size 1

base_path="yi/yi.jsonl"
output_path="yi/yi_m_yi.jsonl"
CUDA_VISIBLE_DEVICES=0 python vllm_modify.py \
    --data_file $base_path --output_file $output_path --base_model $MODEL \
    --tensor_parallel_size 1

log_file="yi/yi.log"
ft_path="yi/yi_m_yi.jsonl"
base_path="yi/yi.jsonl"
CUDA_VISIBLE_DEVICES=0 python get_loglikehood.py --log_file $log_file \
    --base_path $base_path --ft_path $ft_path --base_model $MODEL

