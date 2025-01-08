# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class SentenceDataset(Dataset):
    def __init__(self, pd_df):
        self.data_set = pd_df
        self.bert_tokenizer = tokenizer

    def __getitem__(self, index):
        inputs, label = self.data_set[index][0], int(self.data_set[index][1])
        return inputs['input_ids'], inputs['attention_mask'], inputs['syntax_mask'], label

    def __len__(self):
        return len(self.data_set)
