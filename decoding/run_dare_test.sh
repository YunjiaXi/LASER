# amazon, user knowledge generation
CUDA_VISIBLE_DEVICES=0 python llama_benchmark_group.py --warmup_count -1 --dataset_name amz-new --max_new_tokens 1024 --data_type history --group=3 --test_group 0  --choose_topk 2 --choose_topp 0.1 --model_name vicuna-7b-v1.3 --framework KAR
# amazon, item knowledge generation
CUDA_VISIBLE_DEVICES=0 python llama_benchmark_group.py --warmup_count -1 --dataset_name amz-new --max_new_tokens 1024 --data_type item --test_group 0  --choose_topk 2 --choose_topp 0.1 --model_name vicuna-7b-v1.3 --framework KAR
# ml-10m, user knowledge generation
#CUDA_VISIBLE_DEVICES=0 python llama_benchmark_group.py --warmup_count -1 --dataset_name ml-10m-new --max_new_tokens 1024 --data_type item --test_group 1  --choose_topk 2 --choose_topp 0.1 --model_name vicuna-7b-v1.3 --framework KAR
# ml-10m, item knowledge generation
#CUDA_VISIBLE_DEVICES=0 python llama_benchmark_group.py --warmup_count -1 --dataset_name ml-10m-new --max_new_tokens 1024 --data_type history --group=3 --test_group 0  --choose_topk 2 --choose_topp 0.1 --model_name vicuna-7b-v1.3 --framework KAR


