import torch
import torch.nn as nn
import random as rd
import sys
sys.path.insert(0, 'networks')
from q_v_transformer import CoAttention
from gcn import AdjLearner, GCN
from block import fusions #pytorch >= 1.1.0

class HGA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, ans_decoder, max_len_v, max_len_q, device):
        """
        Reasoning with Heterogeneous Graph Alignment for Video Question Answering (AAAI20)
        """
        super(HGA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.ans_decoder = ans_decoder
        self.max_len_v = max_len_v
        self.max_len_q = max_len_q
        self.device = device
        hidden_size = vid_encoder.dim_hidden
        input_dropout_p = vid_encoder.input_dropout_p

        self.q_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.v_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.co_attn = CoAttention(
            hidden_size, n_layers=vid_encoder.n_layers, dropout_p=input_dropout_p)

        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=input_dropout_p)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=2,
            dropout=input_dropout_p)

        self.gcn_atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-1)) #dim=-2 for attention-pooling otherwise sum-pooling

        self.global_fusion = fusions.Block(
            [hidden_size, hidden_size], hidden_size, dropout_input=input_dropout_p)

        self.fusion = fusions.Block([hidden_size, hidden_size], hidden_size)


    def forward(self, vid_feats, qns, qns_lengths, ans, ans_lengths, teacher_force_ratio=0.5, mode='train'):
        """

        """
        encoder_out, qns_hidden, qns_out, vid_out = self.vq_encoder(vid_feats, qns, qns_lengths)

        batch_size = encoder_out.shape[0]

        hidden = encoder_out.unsqueeze(0)
        if mode == 'train':
            vocab_size = self.ans_decoder.vocab_size
            ans_len = ans.shape[1]
            input = ans[:, 0]
            outputs = torch.zeros(batch_size, ans_len, vocab_size).to(self.device)
            for t in range(0, ans_len):

                output, hidden = self.ans_decoder(qns_out, hidden, input) #attqns, attvid
                outputs[:, t] = output
                teacher_force = rd.random() < teacher_force_ratio
                top1 = output.argmax(1)
                input = ans[:, t] if teacher_force else top1
        else:
            start = torch.LongTensor([1] * batch_size).to(self.device)

            outputs = self.ans_decoder.sample(qns_out, hidden, start) #vidatt, qns_att

        return outputs


    def vq_encoder(self, vid_feats, qns, qns_lengths):
        """

        :param vid_feats:
        :param qns:
        :param qns_lengths:
        :return:
        """
        q_output, s_hidden = self.qns_encoder(qns, qns_lengths)
        qns_last_hidden = torch.squeeze(s_hidden)


        v_output, v_hidden = self.vid_encoder(vid_feats)
        vid_last_hidden = torch.squeeze(v_hidden)

        q_output = self.q_input_ln(q_output)
        v_output = self.v_input_ln(v_output)

        q_output, v_output = self.co_attn(q_output, v_output)

        ### GCN
        adj = self.adj_learner(q_output, v_output)
        # q_v_inputs of shape (batch_size, q_v_len, hidden_size)
        q_v_inputs = torch.cat((q_output, v_output), dim=1)
        # q_v_output of shape (batch_size, q_v_len, hidden_size)
        q_v_output = self.gcn(q_v_inputs, adj)

        ## attention pool
        local_attn = self.gcn_atten_pool(q_v_output)
        local_out = torch.sum(q_v_output * local_attn, dim=1)

        # print(qns_last_hidden.shape, vid_last_hidden.shape)
        global_out = self.global_fusion((qns_last_hidden, vid_last_hidden))


        out = self.fusion((global_out, local_out)).squeeze() #4 x 512

        return out, s_hidden,  q_output, v_output,

