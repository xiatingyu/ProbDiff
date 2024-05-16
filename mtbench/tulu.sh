export RAY_memory_monitor_refresh_ms=0

test_path="data/question.jsonl"
MODEL="../model/tulu-2-dpo-70b"
output_path="tulu/mt/tulu.jsonl"
CUDA_VISIBLE_DEVICES=0,1 python vllm_inference_mt.py \
        --data_file $test_path --output_file $output_path --base_model $MODEL \
        --tensor_parallel_size 2

base_path="tulu/mt/tulu.jsonl"
output_path="tulu/mt/tulu_m_tulu.jsonl"
CUDA_VISIBLE_DEVICES=0,1 python vllm_modify_mt.py \
    --data_file $base_path --output_file $output_path --base_model $MODEL     \
    --tensor_parallel_size 2   

log_file="tulu/mt/tulu.log"
ft_path="tulu/mt/tulu_m_tulu.jsonl"
base_path="tulu/mt/tulu.jsonl"

CUDA_VISIBLE_DEVICES=0,1 python get_loglikehood_mt.py --log_file $log_file \
    --base_path $base_path --ft_path $ft_path --base_model $MODEL


