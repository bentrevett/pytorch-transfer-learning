import torch
import json
from tqdm import tqdm
from collections import defaultdict, Counter
from vocab import Vocab


def get_freqs(path, fields):
    """
    freqs is a dictionary, key being the field name and value
    being a counter for the frequency of each token
    """

    assert isinstance(fields, (list, tuple))

    freqs = defaultdict(Counter)

    with open(path, 'r') as f:
        for line in tqdm(f, desc='Building vocab frequencies...'):
            example = json.loads(line)
            for field in fields:
                freqs[field].update(example[field])

    return freqs


yelp_freqs = get_freqs('data/yelp_train.jsonl', ['tokens', 'tags'])

yelp_tokens_vocab = Vocab(yelp_freqs['tokens'], max_size=25_000,
                          min_freq=2, special_tokens=['<sos>', '<eos>', '<mask>'])

yelp_tags_vocab = Vocab(yelp_freqs['tags'], unk_token=None,
                        special_tokens=['<sos>', '<eos>'])

tokenizer = torch.load('tokenizer_novocab.pt')

tokenizer.vocabs['tokens'] = yelp_tokens_vocab
tokenizer.vocabs['tags'] = yelp_tags_vocab

tokenizer = torch.save(tokenizer, 'tokenizer.pt')
