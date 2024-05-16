MODEL="../model/Qwen-14B-chat"
GPU_index=1

test_path="data/cnn/test.json"
output_path="cnn/14/qwen.jsonl"
CUDA_VISIBLE_DEVICES=$GPU_index python vllm_inference.py \
        --data_file $test_path --output_file $output_path --base_model $MODEL

base_path="cnn/14/qwen.jsonl"
output_path="cnn/14/qwen_m_qwen.jsonl"
CUDA_VISIBLE_DEVICES=$GPU_index python vllm_modify.py \
    --data_file $base_path --output_file $output_path --base_model $MODEL

log_file="cnn/14/0.log"
ft_path="cnn/14/qwen_m_qwen.jsonl"
base_path="cnn/14/qwen.jsonl"

CUDA_VISIBLE_DEVICES=$GPU_index python get_loglikehood.py --log_file $log_file \
    --base_path $base_path --ft_path $ft_path --base_model $MODEL