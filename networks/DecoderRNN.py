import numpy as np
import torch
import torch.nn as nn
from .Attention import TempAttention


class AnsUATT(nn.Module):
    """
    """
    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 glove_embed,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.2,
                 rnn_dropout_p=0):
        super(AnsUATT, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.vocab_size = vocab_size


        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.glove_embed = glove_embed

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(vocab_size, dim_word)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(self.dim_word, self.dim_hidden, n_layers,
            batch_first=True, dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, vocab_size)

        self._init_weights()


    def forward(self, hidden, ans):
        """
         decode answer
        :param encoder_outputs:
        :param encoder_hidden:
        :param ans:
        :param t:
        :return:
        """
        ans_embed = self.embedding(ans)
        ans_embed = self.input_dropout(ans_embed)
        # ans_embed = self.transform(ans_embed)
        ans_embed = ans_embed.unsqueeze(1)
        decoder_input = ans_embed
        outputs, hidden = self.rnn(decoder_input, hidden)
        final_outputs = self.out(outputs.squeeze(1))
        return final_outputs, hidden


    def sample(self, hidden, start=None):
        """

        :param encoder_outputs:
        :param encoder_hidden:
        :return:
        """
        sample_ids = []
        start_embed = self.embedding(start)
        start_embed = self.input_dropout(start_embed)
        # start_embed = self.transform(start_embed)
        start_embed = start_embed.unsqueeze(1)

        inputs = start_embed

        for i in range(self.max_length):
            outputs, hidden = self.rnn(inputs, hidden)
            outputs = self.out(outputs.squeeze(1))
            _, predict = outputs.max(1)
            sample_ids.append(predict)
            ans_embed = self.embedding(predict)
            ans_embed = self.input_dropout(ans_embed)
            # ans_embed = self.transform(ans_embed)

            inputs = ans_embed.unsqueeze(1)

        sample_ids = torch.stack(sample_ids, 1)
        return sample_ids


    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)
        glove_embed = np.load(self.glove_embed)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(glove_embed))



class AnsHME(nn.Module):
    """
    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 glove_embed,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.2,
                 rnn_dropout_p=0):
        super(AnsHME, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.vocab_size = vocab_size


        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.glove_embed = glove_embed

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(vocab_size, dim_word)
        word_mat = torch.FloatTensor(np.load(self.glove_embed))
        self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        self.transform = nn.Sequential(nn.Linear(dim_hidden*2+dim_word, dim_hidden),
                                       nn.Dropout(input_dropout_p))

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(self.dim_hidden, self.dim_hidden, n_layers,
            batch_first=True, dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, vocab_size)

        self._init_weights()


    def forward(self, encoder_outputs, hidden, ans_idx):
        """
         decode answer
        :param encoder_outputs:
        :param encoder_hidden:
        :param ans:
        :param t:
        :return:
        """
        ans_embed = self.embedding(ans_idx)
        ans_embed = torch.cat((encoder_outputs, ans_embed), dim=-1)
        ans_embed = self.transform(ans_embed)
        ans_embed = ans_embed.unsqueeze(1)
        decoder_input = ans_embed
        outputs, hidden = self.rnn(decoder_input, hidden)
        final_outputs = self.out(outputs.squeeze(1))
        return final_outputs, hidden


    def sample(self, encoder_outputs, hidden, start=None):
        """

        :param encoder_outputs:
        :param encoder_hidden:
        :return:
        """
        sample_ids = []
        start_embed = self.embedding(start)
        start_embed = torch.cat((encoder_outputs, start_embed), dim=-1)
        start_embed = self.transform(start_embed)
        start_embed = start_embed.unsqueeze(1)
        inputs = start_embed

        for i in range(self.max_length):
            outputs, hidden = self.rnn(inputs, hidden)
            outputs = self.out(outputs.squeeze(1))
            _, predict = outputs.max(1)
            sample_ids.append(predict)
            ans_embed = self.embedding(predict)
            ans_embed = torch.cat((encoder_outputs, ans_embed), dim=-1)
            ans_embed = self.transform(ans_embed)
            inputs = ans_embed.unsqueeze(1)

        sample_ids = torch.stack(sample_ids, 1)
        return sample_ids


    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)
        nn.init.xavier_normal_(self.transform[0].weight)


class AnsQnsAns(nn.Module):
    """
    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 glove_embed,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.2,
                 rnn_dropout_p=0):
        super(AnsQnsAns, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.vocab_size = vocab_size


        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.glove_embed = glove_embed

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(vocab_size, dim_word)
        word_mat = torch.FloatTensor(np.load(self.glove_embed))
        self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        self.transform = nn.Sequential(nn.Linear(dim_hidden+dim_word, dim_hidden),
                                       nn.Dropout(input_dropout_p))

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(self.dim_hidden, self.dim_hidden, n_layers,
            batch_first=True, dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, vocab_size)

        self._init_weights()


    def forward(self, encoder_outputs, hidden, ans_idx):
        """
        :param encoder_outputs:
        :param encoder_hidden:
        :param ans:
        :param t:
        :return:
        """
        ans_embed = self.embedding(ans_idx)
        ans_embed = torch.cat((encoder_outputs, ans_embed), dim=-1)
        ans_embed = self.transform(ans_embed)
        ans_embed = ans_embed.unsqueeze(1)
        decoder_input = ans_embed
        outputs, hidden = self.rnn(decoder_input, hidden)
        final_outputs = self.out(outputs.squeeze(1))
        return final_outputs, hidden


    def sample(self, encoder_outputs, hidden, start=None):
        """

        :param encoder_outputs:
        :param encoder_hidden:
        :return:
        """
        sample_ids = []
        start_embed = self.embedding(start)
        start_embed = torch.cat((encoder_outputs, start_embed), dim=-1)
        start_embed = self.transform(start_embed)
        start_embed = start_embed.unsqueeze(1)
        inputs = start_embed

        for i in range(self.max_length):
            outputs, hidden = self.rnn(inputs, hidden)
            outputs = self.out(outputs.squeeze(1))
            _, predict = outputs.max(1)
            sample_ids.append(predict)
            ans_embed = self.embedding(predict)
            ans_embed = torch.cat((encoder_outputs, ans_embed), dim=-1)
            ans_embed = self.transform(ans_embed)
            inputs = ans_embed.unsqueeze(1)

        sample_ids = torch.stack(sample_ids, 1)
        return sample_ids


    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)
        nn.init.xavier_normal_(self.transform[0].weight)


class AnsAttSeq(nn.Module):
    """

    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 glove_embed,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.2,
                 rnn_dropout_p=0):
        super(AnsAttSeq, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.vocab_size = vocab_size


        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.glove_embed = glove_embed

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(vocab_size, dim_word)
        word_mat = torch.FloatTensor(np.load(self.glove_embed))
        self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        self.temp = TempAttention(dim_word, dim_hidden, dim_hidden//2)

        self.transform = nn.Sequential(nn.Linear(dim_word+dim_hidden, dim_hidden),
                                       nn.Dropout(input_dropout_p))

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(self.dim_hidden, self.dim_hidden, n_layers,
            batch_first=True, dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, vocab_size)

        self._init_weights()


    def forward(self, seq_outs, hidden, ans_idx):
        """
         decode answer
        :param encoder_outputs:
        :param encoder_hidden:
        :param ans:
        :param t:
        :return:
        """
        ans_embed = self.embedding(ans_idx)
        ans_embed_att, _ = self.temp(ans_embed, seq_outs)
        ans_embed = torch.cat((ans_embed, ans_embed_att), dim=-1)

        ans_embed = self.transform(ans_embed)
        ans_embed = ans_embed.unsqueeze(1)
        decoder_input = ans_embed
        outputs, hidden = self.rnn(decoder_input, hidden)
        final_outputs = self.out(outputs.squeeze(1))
        return final_outputs, hidden


    def sample(self, seq_outs, hidden, start=None):
        """

        :param encoder_outputs:
        :param encoder_hidden:
        :return:
        """
        sample_ids = []
        start_embed = self.embedding(start)
        start_embed_att, _ = self.temp(start_embed, seq_outs)
        start_embed = torch.cat((start_embed, start_embed_att), dim=-1)
        start_embed = self.transform(start_embed)
        start_embed = start_embed.unsqueeze(1)
        inputs = start_embed

        for i in range(self.max_length):
            outputs, hidden = self.rnn(inputs, hidden)
            outputs = self.out(outputs.squeeze(1))
            _, predict = outputs.max(1)
            sample_ids.append(predict)
            ans_embed = self.embedding(predict)
            ans_embed_att, _ = self.temp(ans_embed, seq_outs)
            ans_embed = torch.cat((ans_embed, ans_embed_att), dim=-1)
            ans_embed = self.transform(ans_embed)
            inputs = ans_embed.unsqueeze(1)

        sample_ids = torch.stack(sample_ids, 1)
        return sample_ids


    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)
        nn.init.xavier_normal_(self.transform[0].weight)


class AnsNavieTrans(nn.Module):
    """
    """
    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 glove_embed,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.2,
                 rnn_dropout_p=0):
        super(AnsNavieTrans, self).__init__()

        self.bidirectional_encoder = bidirectional
        self.vocab_size = vocab_size


        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.glove_embed = glove_embed

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(vocab_size, dim_word)
        word_mat = torch.FloatTensor(np.load(self.glove_embed))
        self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        self.transform = nn.Sequential(nn.Linear(dim_word, dim_hidden),
                                       nn.Dropout(input_dropout_p))

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers,
            batch_first=True, dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, vocab_size)

        self._init_weights()


    def forward(self, hidden, ans_idx):
        """
         decode answer
        :param encoder_outputs:
        :param encoder_hidden:
        :param ans:
        :param t:
        :return:
        """
        ans_embed = self.embedding(ans_idx)
        ans_embed = self.transform(ans_embed)
        ans_embed = ans_embed.unsqueeze(1)
        decoder_input = ans_embed
        outputs, hidden = self.rnn(decoder_input, hidden)
        final_outputs = self.out(outputs.squeeze(1))
        return final_outputs, hidden


    def sample(self, hidden, start=None):
        """
        :param encoder_outputs:
        :param encoder_hidden:
        :return:
        """
        sample_ids = []
        start_embed = self.embedding(start)
        start_embed = self.transform(start_embed)
        start_embed = start_embed.unsqueeze(1)
        inputs = start_embed

        for i in range(self.max_length):
            outputs, hidden = self.rnn(inputs, hidden)
            outputs = self.out(outputs.squeeze(1))
            _, predict = outputs.max(1)
            sample_ids.append(predict)
            ans_embed = self.embedding(predict)
            ans_embed = self.transform(ans_embed)
            inputs = ans_embed.unsqueeze(1)

        sample_ids = torch.stack(sample_ids, 1)
        return sample_ids


    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)
        nn.init.xavier_normal_(self.transform[0].weight)


