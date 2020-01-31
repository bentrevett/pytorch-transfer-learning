class Vocab:
    def __init__(self, freqs, max_size=None, min_freq=1,
                 unk_token='<unk>', pad_token='<pad>',
                 special_tokens=[]):

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
            stoi[self.unk_token] = len(itos)
        if self.pad_token is not None:
            stoi[self.pad_token] = len(itos)
        for token in self.special_tokens:
            stoi[token] = len(itos)

        for token, count in freqs.most_common(self.max_size):
            assert token not in itos
            if count < self.min_freq:
                break
            else:
                itos.append(token)

        stoi.update({t: i for i, t in enumerate(itos)})

        return stoi, itos
