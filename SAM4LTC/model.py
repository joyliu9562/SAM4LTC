import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from config import Config

config = Config
tokenizer = AutoTokenizer.from_pretrained(config.model_path)

class BertBasedClassifier(nn.Module):
    def __init__(self):
        super(BertBasedClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(config.model_path)
        self.bert.resize_token_embeddings(len(tokenizer))
        # self.bert.config.dropout_rate = 0
        self.fc = nn.Linear(self.bert.config.hidden_size * 2, config.num_class)
        # self.fc1 = nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, syntax_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        syntax_h = self.entity_average(sequence_output, syntax_mask)
        concat_h = torch.cat([pooled_output, syntax_h], dim=-1)
        logits = self.fc(concat_h)
        return logits, concat_h


    @staticmethod
    def entity_average(hidden_output, syntax_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
            e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        # 加了一个维度
        e_mask_unsqueeze = syntax_mask.unsqueeze(1)
        # 计算每行非0元素的个数
        length_tensor = (syntax_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # bmm函数的作用是对两个三维的张量进行矩阵乘法，也就是批量矩阵乘法 (batch_size, n, m) * (batch_size, m, p) -> (batch_size, n, p)
        # [batch_size, 1, j-i+1] * [batch_size, j-i+1, dim] = [batch_size, 1, dim] -> [batch_size, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector









