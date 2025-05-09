# use BERT/chatGLM to encode the knowledge generated by LLM
import os
import json

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from utils import save_json, get_paragraph_representation
device = 'cuda'


def load_data(path):
    res = []
    # user_set = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if isinstance(data, str):
                # str中存在转义字符时第一次会返回str，需要再处理一遍
                data = json.loads(data)
            res.append([data['user_id'], data['prompt'], data['answer']])
            # user_set.append(data['user_id'])
    # print('42621' in user_set)
    # exit()
        # data = json.load(f)
        # for id, value in data.items():
        #     res.append([id, value['prompt'], value['answer']])
    return res


def get_history_text(data_path, framework):
    raw_data = load_data(data_path)
    idx_list, hist_text = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        if framework in ['KAR', 'TRAWL']:
            prompt = prompt.split('rating history:', 1)[-1]
            pure_hist = prompt[::-1].split(';', 1)[-1][::-1]
            hist_text.append(pure_hist + '. ' + answer)
        elif framework in ['RLMRec']:
            pure_hist = prompt.split('JSON string.', 1)[-1]
            hist_text.append(pure_hist + answer)
        elif framework in ['ONCE']:
            prompt = prompt.split('listed below:', 1)[-1]
            pure_hist = prompt[::-1].split(';', 1)[-1][::-1]
            hist_text.append(pure_hist + '. ' + answer)
        idx_list.append(idx)
    return idx_list, hist_text


def get_item_text(data_path):
    raw_data = load_data(data_path)
    idx_list, text_list = [], []
    for piece in raw_data:
        idx, prompt, answer = piece
        text_list.append(answer)
        idx_list.append(idx)
    return idx_list, text_list


def get_text_data_loader(data_path, batch_size, data_name, framework):
    target_path = os.path.join(data_path, data_name + '.json')
    if 'user' in data_name or 'history' in data_name:
        idxes, text = get_history_text(target_path, framework)
        # print('42621' in idxes)
    else:
        idxes, text = get_item_text(target_path)
    print(data_path, text[0], 'data len', len(text))
    text_loader = DataLoader(text, batch_size, shuffle=False)
    return text_loader, idxes
    # item_idxes, items = get_item_text(os.path.join(data_path, 'item.klg'))
    # print('chatgpt.item 1', items[1], 'item len', len(items))
    #
    # history_loader = DataLoader(history, batch_size, shuffle=False)
    # item_loader = DataLoader(items, batch_size, shuffle=False)
    # return history_loader, hist_idxes, item_loader, item_idxes


def remap_item(item_idxes, item_vec):
    item_vec_map = {}
    for idx, vec in zip(item_idxes, item_vec):
        item_vec_map[idx] = vec
    return item_vec_map


def inference(model, tokenizer, dataloader, model_name, aggregate_type):
    pred_list = []
    model.eval()
    with torch.no_grad():
        for x in tqdm(dataloader):
            torch.cuda.empty_cache()
            if model_name == 'chatglm' or model_name == 'chatglm2':
                x = tokenizer(x, padding=True, truncation=True, return_tensors="pt",
                              return_attention_mask=True).to(device)
                mask = x['attention_mask']
                x.pop('attention_mask')
                outputs = model.transformer(**x, output_hidden_states=True, return_dict=True)
                outputs.last_hidden_state = outputs.last_hidden_state.transpose(1, 0)
            else:
                x = tokenizer(x, padding=True, truncation=True, max_length=512, return_tensors="pt",
                              return_attention_mask=True).to(device)
                mask = x['attention_mask']
                outputs = model(**x, output_hidden_states=True, return_dict=True)
            pred = get_paragraph_representation(outputs, mask, aggregate_type)
            pred_list.extend(pred.tolist())
    return pred_list


def main(knowledge_path, data_path, model_name, batch_size, aggregate_type, data_name, framework, llm_name):
    # hist_loader, hist_idxes, item_loader, item_idxes = get_text_data_loader(knowledge_path, batch_size)
    text_loader, idxes = get_text_data_loader(knowledge_path, batch_size, data_name, framework)

    if model_name == 'chatglm':
        checkpoint = '../../pretrained_models/chatglm-6b' if os.path.exists('../../llm/chatglm-6b') else 'chatglm-6b'
    elif model_name == 'chatglm2':
        checkpoint = '../../pretrained_models/chatglm-v2' if os.path.exists('../../llm/chatglm-v2') else 'chatglm-v2'
    elif model_name == 'bert':
        checkpoint = '../../pretrained_models/bert-base-uncased' \
            if os.path.exists('../../pretrained_models/bert-base-uncased') else 'bert-base-uncased'
    else:
        raise NotImplementedError

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint,  trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint,  trust_remote_code=True).half().cuda()

    # item_vec = inference(model, tokenizer, item_loader, model_name, aggregate_type)
    vec = inference(model, tokenizer, text_loader, model_name, aggregate_type)
    # item_vec_dict = remap_item(item_idxes, item_vec)
    text_vec_dict = remap_item(idxes, vec)

    # print('42621' in text_vec_dict)

    embed_path = os.path.join(data_path, framework, llm_name)
    os.makedirs(embed_path, exist_ok=True)
    # save_json(item_vec_dict, os.path.join(data_path, '{}_{}_augment.item'.format(model_name, aggregate_type)))
    save_json(text_vec_dict, os.path.join(embed_path, '{}_{}_{}_augment.emb'.format(model_name,
                                                                                    aggregate_type,
                                                                                    data_name)))

    stat_path = os.path.join(data_path, 'stat.json')
    with open(stat_path, 'r') as f:
        stat = json.load(f)

    stat['dense_dim'] = 4096 if model_name == 'chatglm' or model_name == 'chatglm2' else 768
    with open(stat_path, 'w') as f:
        stat = json.dumps(stat)
        f.write(stat)


if __name__ == '__main__':
    DATA_DIR = '../data/'
    # DATA_SET_NAME = 'amz-new'
    DATA_SET_NAME = 'ml-10m-new'
    LLM_NAME = 'vicuna-7b-v1.3'
    # FRAMEWORK = 'KAR'
    # FRAMEWORK = 'TRAWL'
    # FRAMEWORK = 'ONCE'
    FRAMEWORK = 'RLMRec'
    DATA_NAME = 'user_short_2'
    KLG_PATH = os.path.join(DATA_DIR, DATA_SET_NAME, 'knowledge', FRAMEWORK, LLM_NAME)
    DATA_PATH = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    # MODEL_NAME = 'chatglm'
    # MODEL_NAME = 'chatglm2'
    MODEL_NAME = 'bert'  # bert, chatglm, chatglm2
    AGGREGATE_TYPE = 'avg'  # last, avg, wavg, cls, ...
    BATCH_SIZE = 64 if MODEL_NAME == 'bert' else 2
    if DATA_SET_NAME == 'amz-new':
        PREFIX_LIST = ['recent_item_ans_0_80000_p0.0_k1', 'recent_history_ans_0_50000_p0.0_k1',
                       'recent_item_ans_0_80000_p0.1_k2', 'recent_history_ans_0_50000_p0.1_k2',]
    elif DATA_SET_NAME == 'ml-10m-new':
        PREFIX_LIST = ['recent_history_ans_0_70000_p0.0_k1', 'recent_item_ans_0_20000_p0.0_k1',
                       'recent_history_ans_0_70000_p0.1_k2', 'recent_item_ans_0_20000_p0.1_k2']
    else:
        raise NotImplementedError

    for DATA_NAME in PREFIX_LIST:
        main(KLG_PATH, DATA_PATH, MODEL_NAME, BATCH_SIZE, AGGREGATE_TYPE, DATA_NAME,
             FRAMEWORK, LLM_NAME)

