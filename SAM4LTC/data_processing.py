# -*- coding: utf-8 -*-
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import json
from utils import *
from config import Config

config = Config

labeled_data = pd.read_pickle(config.data_path).reset_index(drop=True)
tokenizer = AutoTokenizer.from_pretrained(config.model_path)
special_tokens_dict = {'additional_special_tokens': ['[PERSON]', '[TIME]', '[LOC]', '[/PERSON]', '[/TIME]', '[/LOC]']}
label_dic = json.loads(open('./label_dic.json'))


processed_data_set = []
count = 0
for idx, row in labeled_data.iterrows():
    sentence = row['sentence']
    human = row['human']
    time = row['time']
    location = row['location']
    label = label_dic[row['label']]
    doc = nlp(sentence)

    sent_tokens = add_trip_tokens(sentence, human, time, location)
    trip_token = get_trip_token(human, time, location)
    verbs = get_verbs_from_sentence(doc)
    try:
        # 如果sub token获取失败，那么就转到动词和三元组的mask即可
        G = get_graph_of_sentence(doc)
        sub_nodes = get_sub_nodes(G, verbs, trip_token)
        sub_tokens = get_sub_tokens(sub_nodes)
    except Exception as e:
        print(idx, e)
        # print(sentence)
        sub_nodes = []
        sub_nodes.extend(verbs)
        sub_nodes.extend(trip_token)
        sub_tokens = get_sub_tokens(sub_nodes)
        count += 1

    inputs = padding_for_input(sent_tokens, sub_tokens, 128, return_tensor=True)
    processed_data_set.append((inputs, label))
print(count)