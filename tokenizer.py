import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class NLTKTokenizer:
    def __init__(self, lower=True, max_length=250):

        self.lower = lower
        self.max_length = max_length
        self.vocabs = dict()

    def tokenize(self, text):

        assert isinstance(text, str)

        tokens = nltk.word_tokenize(text)

        if self.max_length is not None:
            tokens = tokens[:self.max_length]

        tokens, tags = zip(*nltk.pos_tag(tokens))

        tags = list(tags)

        if self.lower:
            tokens = [token.lower() for token in tokens]

        return tokens, tags

    def numericalize(self, example, vocab_names):

        assert len(example) == len(vocab_names)
        assert all([name in self.vocabs for name in vocab_names])

        for i, field, vocab_name in enumerate(zip(example, vocab_names)):
            if vocab_name is not None:
                vocab = self.vocabs[vocab_name]
                unk_idx = vocab.stoi[vocab.unk_token]
                example[i] = [vocab.stoi.get(t, unk_idx) for t in field]

        return example
