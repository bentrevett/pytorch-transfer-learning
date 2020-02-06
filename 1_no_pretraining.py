import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
import random
import numpy as np

import models

os.makedirs('checkpoints', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)
parser.add_argument('--n_layers', default=3, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--seed', default=1, type=int)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


def load_data(path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line)


def get_data(data, numericalizer, fields):
    _data = []
    for datum in tqdm(data, desc='Creating dataset...'):
        datum_fields = set(datum.keys())
        for field in datum_fields:
            if field not in fields:
                del datum[field]
        _data.append(numericalizer(datum))
    return _data


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, fields):
        self.tokenizer = tokenizer
        self.fields = fields
        data_iter = load_data(path)
        self.data = get_data(data_iter, tokenizer.numericalize, fields)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, batch):

        batch_data = defaultdict(list)

        for item in batch:
            for field in item.keys():
                if field in self.tokenizer.vocabs:
                    batch_data[field].append(torch.LongTensor(item[field]))
                    batch_data[field + '_lengths'].append(len(item[field]))
                else:
                    batch_data[field].append(item[field])

        for field, values in batch_data.items():
            if field in self.tokenizer.vocabs:
                pad_token = self.tokenizer.vocabs[field].pad_token
                pad_idx = self.tokenizer.vocabs[field].stoi[pad_token]
                batch_data[field] = pad_sequence(values,
                                                 padding_value=pad_idx).cuda()
            else:
                batch_data[field] = torch.LongTensor(values).cuda()

        return batch_data


tokenizer = torch.load('tokenizer.pt')

train_data = TextDataset('data/yelp_train.jsonl', tokenizer, ['label', 'tokens'])
test_data = TextDataset('data/yelp_test.jsonl', tokenizer, ['label', 'tokens'])

train_iterator = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                            collate_fn=train_data.collate)
test_iterator = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                           collate_fn=test_data.collate)

vocab_size = len(tokenizer.vocabs['tokens'].itos)
pad_token = tokenizer.vocabs['tokens'].pad_token
pad_idx = tokenizer.vocabs['tokens'].stoi[pad_token]

model = models.BiLSTM(vocab_size, args.embedding_dim, args.hidden_dim,
                      args.n_layers, args.dropout, pad_idx)
head = models.SentimentHead(args.hidden_dim, 2)

model = model.cuda()
head = head.cuda()

optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()),
                       lr=args.lr)

criterion = nn.CrossEntropyLoss()

criterion = criterion.cuda()


def categorical_accuracy(prediction, label):
    max_preds = prediction.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(label)
    return correct.sum() / torch.FloatTensor([label.shape[0]])


def train(model, head, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(iterator, desc='Training...'):

        text = batch['tokens']
        text_lengths = batch['tokens_lengths']
        label = batch['label']

        optimizer.zero_grad()

        output, hidden = model(text, text_lengths)
        prediction = head(output, hidden)

        loss = criterion(prediction, label)

        acc = categorical_accuracy(prediction, label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, head, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in tqdm(iterator, desc='Evaluating...'):

            text = batch['tokens']
            text_lengths = batch['tokens_lengths']
            label = batch['label']

            output, hidden = model(text, text_lengths)
            prediction = head(output, hidden)

            loss = criterion(prediction, label)

            acc = categorical_accuracy(prediction, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


best_valid_loss = float('inf')

for epoch in range(args.n_epochs):

    train_loss, train_acc = train(model, head, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, head, test_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'checkpoints/model-1.pt')
        torch.save(head.state_dict(), 'checkpoints/head-1.pt')

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
