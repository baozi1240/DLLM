data_set=humaneval
gen_len=256
task_index=11
topk=16

# fast
#echo "Fastdllm decoding version..."
#python decode_step_stats.py \
     #--dataset $data_set \
     #--model_path models/Dream-v0-Base-7B-Fastdllm \
     #--block_length 32 \
     #--use_cache \
     #--dual_cache \
     #--max_new_tokens $gen_len \
     #--steps $gen_len \
     #--show_time \
     #--task_index $task_index

# ours
#echo "Refresh First decoding version..."
#python decode_step_stats.py \
     #--dataset $data_set \
     #--model_path models/Dream-v0-Base-7B-Softmax \
     #--block_length 32 \
     #--use_cache \
     #--dual_cache \
     #--focus_decode \
     #--focus_topk $topk \
     #--max_new_tokens $gen_len \
     #--steps $gen_len \
     #--show_time \
     #--task_index $task_index \
     #--gamma 0.1

echo "Optimized version..."
python decode_step_stats.py \
     --dataset $data_set \
     --model_path models/Dream-v0-Base-7B-OptSliding \
     --block_length 32 \
     --use_cache \
     --dual_cache \
     --focus_decode \
     --focus_topk $topk \
     --max_new_tokens $gen_len \
     --steps $gen_len \
     --show_time \
     --task_index $task_index \
     --gamma 0.1

echo "Refresh Parallel decoding version..."
python decode_step_stats.py \
     --dataset $data_set \
     --model_path models/Dream-v0-Base-7B-Sliding \
     --block_length 32 \
     --use_cache \
     --dual_cache \
     --focus_decode \
     --focus_topk $topk \
     --max_new_tokens $gen_len \
     --steps $gen_len \
     --show_time \
     --task_index $task_index \
     --gamma 0.1

echo "Optimized version..."
python decode_step_stats.py \
     --dataset $data_set \
     --model_path models/Dream-v0-Base-7B-OptSliding \
     --block_length 32 \
     --use_cache \
     --dual_cache \
     --focus_decode \
     --focus_topk $topk \
     --max_new_tokens $gen_len \
     --steps $gen_len \
     --show_time \
     --task_index $task_index \
     --gamma 0.1

echo "Optimized Probe Attention version..."
python decode_step_stats.py \
     --dataset $data_set \
     --model_path models/Dream-v0-Base-7B-ProbeOptSliding \
     --block_length 32 \
     --use_cache \
     --dual_cache \
     --focus_decode \
     --focus_topk $topk \
     --max_new_tokens $gen_len \
     --steps $gen_len \
     --show_time \
     --task_index $task_index \
     --gamma 0.1