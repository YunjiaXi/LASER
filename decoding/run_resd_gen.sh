# amazon, user knowledge generation
CUDA_VISIBLE_DEVICES=0 python llama_generate_answer.py --start_id 0 --end_id 70000 --file_name recent_history --dataset_name amz-new --type ans --framework KAR --model_name vicuna-7b-v1.3
# amazon, item knowledge generation
CUDA_VISIBLE_DEVICES=0 python llama_generate_answer.py --start_id 0 --end_id 40000 --file_name recent_item --dataset_name amz-new --type ans --framework KAR --model_name vicuna-7b-v1.3
# ml-10m, user knowledge generation
#CUDA_VISIBLE_DEVICES=0 python llama_generate_answer.py --start_id 0 --end_id 50000 --file_name recent_history --dataset_name ml-10m-new --type ans --framework KAR --model_name vicuna-7b-v1.3 KAR
# ml-10m, item knowledge generation
#CUDA_VISIBLE_DEVICES=0 python llama_generate_answer.py --start_id 0 --end_id 10000 --file_name recent_item --dataset_name ml-10m-new --type ans --framework KAR --model_name vicuna-7b-v1.3


