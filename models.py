import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers,
                 dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=True,
                           dropout=0 if n_layers < 2 else dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        # text = [text len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [text len, batch size, emb dim]

        # pack sequence
        packed_embedded = pack_padded_sequence(embedded, text_lengths,
                                               enforce_sorted=False)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = pad_packed_sequence(packed_output)

        # output = [text len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        output = self.dropout(output)
        hidden = self.dropout(hidden)

        return output, hidden


class SentimentHead(nn.Module):

    def __init__(self, hidden_dim, output_dim):

        super().__init__()

        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, output, hidden):

        # output = [text len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        prediction = self.fc_out(hidden)

        # prediction = [batch size, out dim]

        return prediction


class TagHead(nn.Module):

    def __init__(self, hidden_dim, output_dim):

        super().__init__()

        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, output, hidden):

        # output = [text len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        predictions = self.fc_out(output)

        # predictions = [text len, batch size, out dim]

        return predictions
