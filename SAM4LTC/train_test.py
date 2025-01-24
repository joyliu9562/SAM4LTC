# -*- coding: utf-8 -*-

from data_processing import processed_data_set
from utils import seed_everything, contrastive_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from dataloader import SentenceDataset
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from model import BertBasedClassifier
import torch
import torch.optim as optim
import wanbd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kf = KFold(n_splits=10, random_state=42, shuffle=True)
epochs = 20

all_pred = {}
all_label = {}
all_stat = []

print('start training。。。。。。。。。。。')
# for random_seed in [88, 89, 99, 45, 79, 49]:
for fold, (train_idx, val_idx) in enumerate(kf.split(processed_data_set)):
    print('fold', fold)
    # X_train, X_test = get_train_test_df(labeled_data, random_seed)
    # X_train, X_test = train_test_split(processed_data_set, test_size=0.2, random_state=random_seed)
    seed_everything(42)
    X_train = [processed_data_set[i] for i in train_idx]
    X_test = [processed_data_set[i] for i in val_idx]

    train_set = SentenceDataset(X_train)
    test_set = SentenceDataset(X_test)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model = BertBasedClassifier()
    model.to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    pre_score = 0
    best_score, best_precision, best_recall, best_f1 = None, None, None, None

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        # print("Epoch : #", epoch + 1)
        for idx, row in enumerate(train_loader):
            input_ids = row[0].squeeze(1).to(device)
            # attention_mask = row[1].squeeze(1).to(device)
            origin_mask = row[1].squeeze(1).to(device)
            syntax_mask = row[2].squeeze(1).to(device)
            y = row[3].to(device)
            bert_output, bert_embedding = model(input_ids, origin_mask, syntax_mask)

            try:
                contrast_loss = contrastive_loss(0.1, bert_embedding.cpu().detach().numpy(), y)
            except Exception as e:
                print(idx, str(e))
                print(bert_embedding)
                break

            ce_loss = criterion(bert_output, y)
            # loss = 0.7 * contrast_loss + 0.3 * ce_loss
            # loss = ce_loss

            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            train_loss += ce_loss.item() / len(y)
        # print("Loss: ", train_loss)
        model.eval()
        val_loss = 0
        score = None
        precision, recall, f1 = None, None, None
        total_pred = []
        total_labels = []
        total_source = []
        for idx, row in enumerate(test_loader):
            input_ids = row[0].squeeze(1).to(device)
            # attention_mask = row[1].squeeze(1).to(device)
            origin_mask = row[1].squeeze(1).to(device)
            syntax_mask = row[2].squeeze(1).to(device)
            y = row[3].to(device)
            bert_output, _ = model(input_ids, origin_mask, syntax_mask)
            _, predicted = torch.max(bert_output.data, 1)
            total_pred += list(predicted.cpu().numpy().reshape(-1))
            total_labels += list(y.cpu().numpy().reshape(-1))

        score = accuracy_score(total_labels, total_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(total_labels, total_pred, average='weighted')

        if score >= pre_score:
            pre_score = score
            all_label[fold] = total_labels
            all_pred[fold] = total_pred
            precision, recall, f1, _ = precision_recall_fscore_support(total_labels, total_pred, average='weighted')
            best_score, best_precision, best_recall, best_f1 = score, precision, recall, f1
        # print(score)
    print(fold, best_score, best_precision, best_recall, best_f1)
    all_stat.append(
        {'random_seed': fold, 'best_score': best_score, 'best_precision': best_precision, 'best_recall': best_recall,
         'best_f1': best_f1})


