# amazon, user knowledge generation
CUDA_VISIBLE_DEVICES=0 python llama_generate_answer.py --start_id 0 --end_id 50000 --file_name history --dataset_name amz-new --type warmup --model_name vicuna-7b-v1.3 --framework KAR
# amazon, item knowledge generation
CUDA_VISIBLE_DEVICES=0 python llama_generate_answer.py --start_id 0 --end_id 40000 --file_name item --dataset_name amz-new --type warmup --model_name vicuna-7b-v1.3 --framework KAR
# ml-10m, user knowledge generation
#CUDA_VISIBLE_DEVICES=0 python llama_generate_answer.py --start_id 0 --end_id 70000 --file_name history --dataset_name ml-10m-new --type warmup --model_name vicuna-7b-v1.3 --framework KAR
# ml-10m, items knowledge generation
#CUDA_VISIBLE_DEVICES=0 python llama_generate_answer.py --start_id 0 --end_id 10000 --file_name item --dataset_name ml-10m-new --type warmup --model_name vicuna-7b-v1.3 --framework KAR
