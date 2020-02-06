from tqdm import tqdm
from collections import Counter


class Vocab:
    def __init__(self, freqs, max_size=None, min_freq=1,
                 unk_token='<unk>', pad_token='<pad>',
                 special_tokens=[]):

        assert isinstance(freqs, Counter)

        self.max_size = max_size
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens

        self.stoi, self.itos = self.create_vocab(freqs)

    def create_vocab(self, freqs):

        stoi = dict()
        itos = list()

        if self.unk_token is not None:
            itos.append(self.unk_token)
        if self.pad_token is not None:
            itos.append(self.pad_token)
        for token in self.special_tokens:
            itos.append(token)

        for token, count in tqdm(freqs.most_common(self.max_size), desc='Creating vocab...'):
            if token in itos:
                print(f'Tried to add {token}, which is already in vocab')
                continue
            if count < self.min_freq:
                break
            else:
                itos.append(token)

        stoi.update({t: i for i, t in enumerate(itos)})

        return stoi, itos
