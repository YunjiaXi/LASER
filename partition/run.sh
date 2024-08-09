# for amz
dataset_name='amz-new'
# for ml-10m
#dataset_name='ml-10m-new'

python train_encoder.py --model lightgcn --dataset dataset_name --cuda 0
python cluster_data.py --dataset=dataset_name
