import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm
from collections import defaultdict


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
