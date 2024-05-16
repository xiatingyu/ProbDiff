
MODEL="model/Qwen-14B-chat"
GPU_index=0
test_path="data/data_release.jsonl"
output_path="align/ft.jsonl"
CUDA_VISIBLE_DEVICES=$GPU_index python vllm_inference.py \
        --data_file $test_path --output_file $output_path --base_model $MODEL


MODEL="model/Qwen-14B-chat"
GPU_index=0
base_path="align/qwen.jsonl"
output_path="align/qwen_m_qwen.jsonl"
CUDA_VISIBLE_DEVICES=$GPU_index python vllm_modify.py \
    --data_file $base_path --output_file $output_path --base_model $MODEL 


MODEL="model/Qwen-14B-chat"
GPU_index=0
log_file="align/qwen_m_qwen.log"
ft_path="align/qwen_m_qwen.jsonl"
base_path="align/qwen.jsonl"
CUDA_VISIBLE_DEVICES=$GPU_index python get_loglikehood.py --log_file $log_file \
    --base_path $base_path --ft_path $ft_path --base_model $MODEL
