# Efficiency Unleashed: Lossless Acceleration for LLM-based Recommender Systems with Speculative Decoding (LASER)

Our paper was accepted by SIGIR 2025!

## Requirements

```

python>=3.8
torch>=2.2.1
transformers==4.38.2
numpy
scikit-learn
tiktoken
einops
pandas
```

## Setup

1. Download datasets & LLMs

   Take [Amazon-Books](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) for example, download the dataset to folder `data/amz-new/raw_data/`

   Download LLMs, such as [vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3), to the floder `pretrained_models`
2. Preprocessing: in folder `preprocess`

   1. run `python preprocess_amz.py` to preprocess Amazon dataset.
   2. run `python generate_data_and_prompt.py` to generate data for CTR and prompt for LLMs, as well as item grouping (attribute-based).
   3. run `python generate_lightgcn_data.py` to generate data for lightGCN.
3. Item and user grouping: in folder `partition`

   run `sh run.sh` to train lightGCN and clustering user (collaborative-based).
4. Knowledge generation: in folder `decoding`(where the accelaration happens)

   1. run `sh run_warmup.sh` to generate warmup/old knowledge for constructing retrieval pool.
   2. run `sh run_dare_tesh.sh` to test the accelaration performance of our proposed DARE.
   3. run `sh run_dare_gen.sh` to generate knowledge for the whole dataset with DARE.
   4. run `sh run_resd_gen.sh` to generate knowledge for the whole dataset with greedy verification (optional).
5. Knowledge encoding: in folder `knowledge_encoding`

   1. Run `python merge_item.py` to merge old and new items
   2. Run `python lm_encoding.py` to encode knowledge
6. Downstream CTR task: in folder `RS`

   Run `python run_ctr.py` for ctr task
