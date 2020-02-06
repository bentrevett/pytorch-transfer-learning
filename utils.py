import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm
from collections import defaultdict
import random


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


def categorical_accuracy(prediction, label):
    max_preds = prediction.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(label)
    return correct.sum() / torch.FloatTensor([label.shape[0]])


def categorical_tag_accuracy(predictions, tags, tag_pad_idx):
    max_preds = predictions.argmax(dim=1, keepdim=True)
    non_pad_elements = (tags != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(tags[non_pad_elements])
    return correct.sum() / torch.FloatTensor([tags[non_pad_elements].shape[0]])


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


class MaskDataset(Dataset):
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
                    assert '<mask>' in self.tokenizer.vocabs[field].stoi
                    mask_idx = self.tokenizer.vocabs[field].stoi['<mask>']
                    mask_pos = random.randint(0, len(item[field])-1)
                    masked_token = item[field][mask_pos]
                    item[field][mask_pos] = mask_idx
                    batch_data[field].append(torch.LongTensor(item[field]))
                    batch_data[field + '_lengths'].append(len(item[field]))
                    batch_data[field + '_mask'].append(masked_token)
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
