import os
import csv
import json
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext.datasets import text_classification
from tqdm import tqdm
from tokenizer import NLTKTokenizer

os.makedirs('data', exist_ok=True)


def load_csv(path, tokenize_fn):
    """
    Yields iterator of tokenized and tagged data
    """

    with open(path, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            label = int(row[0]) - 1
            tokens, tags = tokenize_fn(' '.join(row[1:]))
            yield label, tokens, tags


def get_dataset(name, tokenize_fn, root='data'):
    """
    Downloads and extracts dataset
    Gets iterators over dataset
    """

    dataset_tar = download_from_url(text_classification.URLS[name], root=root)
    extracted_files = extract_archive(dataset_tar)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    train_iterator = load_csv(train_csv_path, tokenize_fn)
    test_iterator = load_csv(test_csv_path, tokenize_fn)

    return train_iterator, test_iterator


def save_data(path, data, field_names):
    """
    Saves data List[Tuple] to jsonlines format
    """

    with open(path, 'w+') as f:
        for example in tqdm(data):
            assert len(example) == len(field_names)
            _example = dict()
            for field, name in zip(example, field_names):
                _example[name] = field
            json.dump(_example, f)
            f.write('\n')


tokenizer = NLTKTokenizer(lower=True,
                          max_length=250)

yelp_train, yelp_test = get_dataset('YelpReviewPolarity', tokenizer.tokenize)

save_data('data/yelp_train.jsonl', yelp_train, ['label', 'tokens', 'tags'])
save_data('data/yelp_test.jsonl', yelp_test, ['label', 'tokens', 'tags'])

amazon_train, amazon_test = get_dataset('AmazonReviewPolarity',
                                        tokenizer.tokenize)

save_data('data/amazon_train.jsonl', amazon_train, ['label', 'tokens', 'tags'])
save_data('data/amazon_test.jsonl', amazon_test, ['label', 'tokens', 'tags'])

torch.save(tokenizer, 'tokenizer_novocab.pt')
