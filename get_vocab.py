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

    freqs = defaultdict(Counter)

    with open(path, 'r') as f:
        for line in tqdm(f):
            example = json.loads(line)
            for field in fields:
                freqs[field].update(example[field])

    return freqs


amazon_freqs = get_freqs('data/amazon_train.jsonl', ['tokens', 'tags'])

amazon_tokens_vocab = Vocab(amazon_freqs['tokens'], max_size=25_000,
                            min_freq=1, special_tokens=['<sos>', '<eos>'])

amazon_tags_vocab = Vocab(amazon_freqs['tags'], unk_token=None,
                          special_tokens=['<sos>', '<eos>'])

tokenizer = torch.load('tokenizer_novocab.pt')

tokenizer.vocabs['tokens'] = amazon_tokens_vocab
tokenizer.vocabs['tags'] = amazon_tags_vocab

tokenizer = torch.save(tokenizer, 'tokenizer.pt')
