import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class NLTKTokenizer:
    def __init__(self, lower=True, max_length=250, sos_token=None,
                 eos_token=None):

        self.lower = lower
        self.max_length = max_length
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.vocabs = dict()

    def tokenize(self, text):
        """
        Tokenizes and tags an input string
        """

        assert isinstance(text, str)

        tokens = nltk.word_tokenize(text)

        if self.max_length is not None:
            tokens = tokens[:self.max_length]

        tokens, tags = zip(*nltk.pos_tag(tokens))

        tags = list(tags)

        if self.lower:
            tokens = [token.lower() for token in tokens]

        if self.sos_token is not None:
            tokens = [self.sos_token] + tokens
            tags = [self.sos_token] + tags

        if self.eos_token is not None:
            tokens = tokens + [self.eos_token]
            tags = tags + [self.eos_token]

        return tokens, tags

    def numericalize(self, example, vocab_names):
        """
        Takes a list of tokens and a vocabulary, numericalizes
        """

        assert not isinstance(example, str)
        assert len(example) == len(vocab_names)
        assert all([name in self.vocabs for name in vocab_names])

        example = list(example)

        for i, (field, vocab_name) in enumerate(zip(example, vocab_names)):
            if vocab_name is not None:
                vocab = self.vocabs[vocab_name]
                if vocab.unk_token is not None:
                    unk_idx = vocab.stoi[vocab.unk_token]
                    example[i] = [vocab.stoi.get(t, unk_idx) for t in field]
                else:
                    example[i] = [vocab.stoi[t] for t in field]

        return example
