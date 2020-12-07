import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import jieba
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from transformers import BertModel,BertTokenizer,BertForNextSentencePrediction,XLNetModel,XLNetTokenizer
from transformers import DistilBertTokenizerFast, DistilBertModel
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelWithLMHead, get_linear_schedule_with_warmup
import torch.nn.functional as F
import random
from sklearn.model_selection import KFold
import datetime
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/text_experiment')
'''
使用具有NSP任务的预训练模型
'''
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

np.random.seed(seed) # Numpy module.
random.seed(seed) # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MyModel(nn.Module):
    def __init__(self, model_name, max_seq_len, hidden_size, n_class):
        super(MyModel, self).__init__()
        self.bert_nsp = BertForNextSentencePrediction.from_pretrained(model_name, return_dict=True)
    def forward(self, x, mask, token_type_ids):
        outputs = self.bert_nsp(x, attention_mask = mask, token_type_ids=token_type_ids)
        outputs = outputs.logits
        return outputs

class Bert_Fc(nn.Module):
    def __init__(self, model_name, max_seq_len, hidden_size, n_class):
        super(Bert_Fc, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)
        self.maxpool = nn.MaxPool1d(max_seq_len)
        self.avgpool = nn.AvgPool1d(max_seq_len)
        self.linear1 = nn.Linear(3 * hidden_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, n_class) 
        self.tanh = nn.Tanh()       
    def forward(self, x, mask, token_type_ids):
        bert_out = self.bert(x, attention_mask = mask, token_type_ids=token_type_ids)
        sentence_emb = bert_out.last_hidden_state[:, 0, :]
        bert_out = bert_out.last_hidden_state.permute(0, 2, 1).contiguous()
        maxpool_out = self.maxpool(bert_out).squeeze(2)
        avgpool_out = self.avgpool(bert_out).squeeze(2)
        outputs = torch.cat((maxpool_out, avgpool_out, sentence_emb), 1)
        outputs = self.tanh(self.linear1(outputs))
        outputs = self.linear2(outputs)
        return outputs
        
class zyDataset(Dataset):
    def __init__(self, encodings, labels, test=False):
        self.encodings = encodings
        self.len = len(self.encodings['input_ids'])
        self.labels = labels
        self.test = test
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if not self.test:
            item['labels'] = torch.tensor(self.labels['label'].tolist()[idx])
        return item

    def __len__(self):
        return self.len

def train(model, optimizer, scheduler, loss_func, train_iter, epoch):
    model.train()
    total_loss = 0
    for idx, tr_batch in enumerate(train_iter):
        input_ids = tr_batch['input_ids'].to(device)
        token_type_ids = tr_batch['token_type_ids'].to(device)
        attention_mask = tr_batch['attention_mask'].to(device)
        label = tr_batch['labels'].to(device)
        
        optimizer.zero_grad()
        output = model(input_ids, attention_mask, token_type_ids)
        
        loss = loss_func(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        log_inte = 300
        if idx % log_inte == 0 and idx != 0:
            # writer.add_scalar('training loss',
            #                 total_loss / log_inte,
            #                 epoch * len(train_iter) + idx)
            print('train-loss: {0:>6.5}'.format(total_loss / log_inte))
            total_loss = 0
    return model
def evalate(model, val_iter):
    model.eval()
    res = []
    labels = []
    total_loss = 0
    with torch.no_grad():
        for idx, val_batch in enumerate(val_iter):
            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            token_type_ids = val_batch['token_type_ids'].to(device)
            label = val_batch['labels'].to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(output, label)
            total_loss += loss.item()
            output = torch.argmax(output, axis=1).tolist()
            label = label.tolist()
            labels += label
            res += output
    res = np.array(res)
    labels = np.array(labels)
    accuracy = metrics.accuracy_score(res, labels)
    precision = metrics.precision_score(res, labels)
    recall = metrics.recall_score(res, labels)
    f1 = metrics.f1_score(res, labels)
    print("Val Acc: {0:>6.5}, Val Pre: {0:>6.5}, Val Rec: {0:>6.5}, F1: {0:>6.5}".format(accuracy, precision, recall, f1))
    return total_loss / len(val_iter), f1

def test(model, test_iter):
    res = []
    # model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        for idx, te_batch in enumerate(test_iter):
            input_ids = te_batch['input_ids'].to(device)
            attention_mask = te_batch['attention_mask'].to(device)
            token_type_ids = te_batch['token_type_ids'].to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            # output = torch.argmax(output, axis=1).tolist()
            output = output.tolist()
            res += output
    return np.array(res)
def get_model_feature(train_iter, test_iter, bs):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    train_feature = []
    test_feature = []
    labels = []
    with torch.no_grad():

        for idx, val_batch in enumerate(train_iter):
            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            token_type_ids = val_batch['token_type_ids'].to(device)
            label = val_batch['labels'].to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            train_feature.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
        for idx, test_batch in enumerate(test_iter):
            input_ids = test_batch['input_ids'].to(device)
            attention_mask = test_batch['attention_mask'].to(device)
            token_type_ids = test_batch['token_type_ids'].to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            test_feature.append(output.cpu().numpy())
    train_feature = np.concatenate(train_feature, axis=0)
    test_feature = np.concatenate(test_feature, axis=0)
    labels = np.concatenate(labels, axis=0)
    return train_feature, labels, test_feature
if __name__ == '__main__':
    """
    目前已经跑完的模型'chinese-bert-wwm-ext','chinese_roberta_wwm_large_ext','roberta-zh-large','chinese-xlnet-mid','chinese-xlnet-base','nezha-large'
    """
    start = datetime.datetime.now()
    base_path = "/home/penglu/Competition/2020House/Q-A-matching-of-real-estate-industry-main"
    max_seq_len = 100
    hidden_size = 1024
    n_class = 2
    batch_size = 8
    model_name = r"/home/penglu/Competition/2020House/Q-A-matching-of-real-estate-industry-main/bert_pretrain/hfl/nezha-large/"

    save_model_dir = base_path + '/save_model'
    epochs = 5
    lr = 2e-5
    min_loss = float('inf')
    last_imporve = 0
    total_batch = 0
    early_stop = 20

    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')

    feature_cols = ['query' ,'reply']
    label_cols = ['label']
    tokenizer = BertTokenizer.from_pretrained(model_name)
    kf = KFold(n_splits=5)
    res_proba = np.zeros((len(test_data), 2))

    for tr_idx, val_idx in kf.split(train_data):
        train_x, train_y = train_data[feature_cols].loc[tr_idx], train_data[label_cols].loc[tr_idx]
        val_x, val_y = train_data[feature_cols].loc[val_idx], train_data[label_cols].loc[val_idx]
        train_encodings = tokenizer(train_x['query'].tolist(), train_x['reply'].tolist(), truncation=True, padding=True, max_length=max_seq_len)
        val_encodings = tokenizer(val_x['query'].tolist(), val_x['reply'].tolist(), truncation=True, padding=True, max_length=max_seq_len)
        test_encodings = tokenizer(test_data['query'].tolist(), test_data['reply'].tolist(), truncation=True, padding=True, max_length=max_seq_len)

        train_dataset = zyDataset(train_encodings, train_y[label_cols].reset_index(drop=True))
        val_dataset = zyDataset(val_encodings, val_y[label_cols].reset_index(drop=True))
        test_dataset = zyDataset(test_encodings, None, test=True)

        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_training_steps = len(train_iter) * epochs

        model = Bert_Fc(model_name, max_seq_len, hidden_size, n_class)
        model.to(device)

        loss_func = nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        save_path = save_model_dir + '/bert_nezha-large_classifier.ckpt'
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
        )
        for epoch in range(1, epochs + 1):
            print('[{}/{}]'.format(epoch, epochs))
            model = train(model, optimizer, scheduler, loss_func, train_iter, epoch)
            val_loss, f1 = evalate(model, val_iter)
            print('test-loss: {0:>6.5}'.format(val_loss))
            total_batch = epoch
            if val_loss < min_loss:
                min_loss = val_loss
                print('saving')
                torch.save(model.state_dict(), save_path)
                last_imporve = total_batch
            if (total_batch - last_imporve) >= early_stop:
                break
        res = test(model, test_iter)
        res_proba += res
        torch.cuda.empty_cache()
    res_proba = res_proba / 5
    res = np.argmax(res_proba, axis=1)
    submit = pd.DataFrame()
    submit['id'] = test_data['id']
    submit['reply_sort'] = test_data['reply_sort']
    submit['label'] = res
    submit.to_csv('./data/submit_cv_nezha-large.tsv', sep='\t', header=False, index=False)
    train_X, train_y = train_data[feature_cols], train_data[label_cols]
    train_encodings = tokenizer(train_X['query'].tolist(), train_X['reply'].tolist(), truncation=True, padding=True,
                                max_length=max_seq_len)
    test_encodings = tokenizer(test_data['query'].tolist(), test_data['reply'].tolist(), truncation=True, padding=True,
                               max_length=max_seq_len)
    # load train datasets
    train_dataset = zyDataset(train_encodings, train_y[label_cols].reset_index(drop=True))
    test_dataset = zyDataset(test_encodings, None, test=True)
    # build dataloader
    train_iter = DataLoader(train_dataset, batch_size=batch_size)
    test_iter = DataLoader(test_dataset, batch_size=batch_size)
    # get the model feautres
    train_feat, labels, test_feat = get_model_feature(train_iter, test_iter, bs=batch_size)
    np.savez('./save_model/nezha-large_train_features.npz',train_feat,labels)
    print('save train_features succeed!!!')
    np.savez('./save_model/nezha-large_test_features.npz',test_feat)
    print('save test_features succeed!!!')
    end = datetime.datetime.now()
    print("Model training time: {}".format(end - start))