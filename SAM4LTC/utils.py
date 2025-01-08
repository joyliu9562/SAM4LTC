# -*- encoding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import networkx as nx
import numpy as np
import spacy
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from config import Config

config = Config

warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained(config.model_path)
nlp = spacy.load('en_core_web_sm')


def get_verbs_from_sentence(doc):
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    return verbs


def get_word_tokens(words: list):
    tokens = []
    for word in words:
        tokens.extend(tokenizer.tokenize(word))
    return tokens


def get_trip_token(human, time, location):
    trip_token = []
    human_doc = nlp(human)
    time_doc = nlp(time)
    location_doc = nlp(location)
    trip_token.extend([token.text for token in human_doc])
    trip_token.extend([token.text for token in time_doc])
    trip_token.extend([token.text for token in location_doc])

    return trip_token


def get_graph_of_sentence(doc):
    G = nx.Graph()
    for token in doc:
        G.add_edge(token.text, token.head.text)
    return G


def get_sub_nodes(G, verbs, trip_token):
    sub_nodes = []
    for source in verbs:
        for target in trip_token:
            sub_nodes.extend(nx.shortest_path(G, source=source, target=target))
    sub_nodes = list(set(sub_nodes))
    if len(sub_nodes) == 0:
        # print('empty')
        return trip_token
    return sub_nodes


def get_sub_tokens(sub_nodes):
    sub_tokens = []
    for sub_node in sub_nodes:
        sub_tokens.extend(tokenizer.tokenize(sub_node))

    # sub_tokens.extend(['[HUMAN]', '[TIME]', '[LOC]', '[/HUMAN]', '[/TIME]', '[LOC]/'])

    return sub_tokens


def add_trip_tokens(sentence, human, time, location):
    sent_tokens = tokenizer.tokenize(sentence)
    human_tokens = tokenizer.tokenize(human)
    # labeled_human_tokens = ['[HUMAN]'] + human_tokens + ['[HUMAN]']
    time_tokens = tokenizer.tokenize(time)
    # labeled_time_tokens = ['[TIME]'] + time_tokens + ['[TIME]']
    location_tokens = tokenizer.tokenize(location)
    # labeled_location = ['[LOC]'] + location_tokens + ['[LOC]']

    human_head_token = human_tokens[0]
    human_head_idx = sent_tokens.index(human_head_token)
    human_tail_token = human_tokens[-1]
    human_tail_idx = sent_tokens.index(human_tail_token)

    sent_tokens.insert(human_head_idx, '[HUMAN]')
    sent_tokens.insert(human_tail_idx + 2, '[/HUMAN]')

    time_head_token = time_tokens[0]
    time_head_idx = sent_tokens.index(time_head_token)
    time_tail_token = time_tokens[-1]
    time_tail_idx = sent_tokens.index(time_tail_token)
    sent_tokens.insert(time_head_idx, '[TIME]')
    sent_tokens.insert(time_tail_idx + 2, '[/TIME]')

    location_head_token = location_tokens[0]
    location_head_idx = sent_tokens.index(location_head_token)
    location_tail_token = location_tokens[-1]
    location_tail_idx = sent_tokens.index(location_tail_token)
    sent_tokens.insert(location_head_idx, '[LOC]')
    sent_tokens.insert(location_tail_idx + 2, '[/LOC]')

    sent_tokens = human_tokens + time_tokens + location_tokens + ['[SEP]'] + sent_tokens

    return sent_tokens


def mask_generation(sent_tokens, sub_tokens):
    mask = [0] * len(sent_tokens)
    for idx, token in enumerate(sent_tokens):
        if token in sub_tokens:
            mask[idx] = 1
    return mask


def padding_for_input(input_tokens, sub_tokens, max_length, return_tensor=False):
    input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
    # input_tokens = [tokenizer.cls_token] + input_tokens
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [1] * len(input_tokens)

    syntax_mask = mask_generation(input_tokens, sub_tokens)

    for i in range(max_length - len(input_tokens)):
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        syntax_mask.append(0)
    is_all_zero = all(x == 0 for x in syntax_mask)
    if is_all_zero:
        syntax_mask = attention_mask.copy()

    if not return_tensor:
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'syntax_mask': syntax_mask}
    else:
        return {'input_ids': torch.tensor([input_ids]), 'attention_mask': torch.tensor([attention_mask]),
                'syntax_mask': torch.tensor([syntax_mask])}


def get_train_test_df(clean_data, random_seed):
    features = ['sentence', 'human', 'time', 'location', 'label']
    target = 'label'
    X_train, X_test, y_train, y_test = train_test_split(
        clean_data[features],
        clean_data[target],
        test_size=0.2,
        random_state=random_seed
    )
    return X_train, X_test


def seed_everything(seed=88):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def contrastive_loss(temp, embedding, label):
    """
    calculate the contrastive loss
    这里的embedding指的是一个batch
    """
    # 首先计算WMD矩阵用来代替余弦相似度矩阵
    # cosine_sim = get_wmd_dist_in_batch(embedding)
    cosine_sim = cosine_similarity(embedding, embedding)
    dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    dis = np.exp(dis)
    cosine_sim = np.exp(cosine_sim)

    row_sum = []
    for i in range(len(embedding)):
        row_sum.append(sum(dis[i]))

    contrastive_loss = 0

    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        inner_sum = 0
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + np.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

