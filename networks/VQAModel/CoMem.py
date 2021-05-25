import torch
import torch.nn as nn
import random as rd
import sys
sys.path.insert(0, 'networks')
from memory_module import EpisodicMemory

class CoMem(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, ans_decoder, max_len_v, max_len_q, device, input_drop_p=0.2):
        """
        motion-appearance co-memory networks for video question answering (CVPR18)
        """
        super(CoMem, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.ans_decoder = ans_decoder

        dim = qns_encoder.dim_hidden

        self.epm_app = EpisodicMemory(dim*2)
        self.epm_mot = EpisodicMemory(dim*2)

        self.linear_ma = nn.Linear(dim*2*3, dim*2)
        self.linear_mb = nn.Linear(dim*2*3, dim*2)

        self.vq2word = nn.Linear(dim*2*2, dim)
        self._init_weights()
        self.device = device

    def _init_weights(self):
        """
        initialize the linear weights
        :return:
        """
        nn.init.xavier_normal_(self.linear_ma.weight)
        nn.init.xavier_normal_(self.linear_mb.weight)
        nn.init.xavier_normal_(self.vq2word.weight)


    def forward(self, vid_feats, qns, qns_lengths, ans, ans_lengths, teacher_force_ratio=0.5, iter_num=3, mode='train'):
        """
        Co-memory network
        """

        outputs_app_l1, outputs_app_l2, outputs_motion_l1, outputs_motion_l2 = self.vid_encoder(vid_feats) #(batch_size, fnum, feat_dim)

        outputs_app = torch.cat((outputs_app_l1, outputs_app_l2), dim=-1)
        outputs_motion = torch.cat((outputs_motion_l1, outputs_motion_l2), dim=-1)

        qns_output, qns_hidden = self.qns_encoder(qns, qns_lengths)

        # qns_output = qns_output.permute(1, 0, 2)
        batch_size, seq_len, qns_feat_dim = qns_output.size()


        qns_embed = qns_hidden.permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)

        m_app = outputs_app[:, -1, :]
        m_mot = outputs_motion[:, -1, :]
        ma, mb = m_app.detach(), m_mot.detach()
        m_app = m_app.unsqueeze(1)
        m_mot = m_mot.unsqueeze(1)
        for _ in range(iter_num):
            mm = ma + mb
            m_app = self.epm_app(outputs_app, mm, m_app)
            m_mot = self.epm_mot(outputs_motion, mm, m_mot)
            ma_q = torch.cat((ma, m_app.squeeze(1), qns_embed), dim=1)
            mb_q = torch.cat((mb, m_mot.squeeze(1), qns_embed), dim=1)
            # print(ma_q.shape)
            ma = torch.tanh(self.linear_ma(ma_q))
            mb = torch.tanh(self.linear_mb(mb_q))

        mem = torch.cat((ma, mb), dim=1)
        encoder_outputs = self.vq2word(mem)
        # hidden = qns_hidden
        hidden = encoder_outputs.unsqueeze(0)

        # decoder_inputs = encoder_outputs

        if mode == 'train':
            vocab_size = self.ans_decoder.vocab_size
            ans_len = ans.shape[1]
            input = ans[:, 0]
            outputs = torch.zeros(batch_size, ans_len, vocab_size).to(self.device)

            for t in range(0, ans_len):
                output, hidden = self.ans_decoder(qns_output, hidden, input)
                outputs[:, t] = output
                teacher_force = rd.random() < teacher_force_ratio
                top1 = output.argmax(1)
                input = ans[:, t] if teacher_force else top1
        else:
            start = torch.LongTensor([1] * batch_size).to(self.device)
            outputs = self.ans_decoder.sample(qns_output, hidden, start)

        return outputs