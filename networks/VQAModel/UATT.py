import torch
import torch.nn as nn
import random as rd
import sys
sys.path.insert(0, 'networks')
from Attention import TempAttentionHis

class UATT(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, ans_decoder, device):
        """
        Unifying the Video and Question Attentions for Open-Ended Video Question Answering (TIP17)
        """
        super(UATT, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.ans_decoder = ans_decoder
        mem_dim = 512
        self.att_q2v = TempAttentionHis(vid_encoder.dim_hidden*2, qns_encoder.dim_hidden*2, mem_dim, mem_dim)
        self.att_v2q = TempAttentionHis(vid_encoder.dim_hidden*2, qns_encoder.dim_hidden*2, mem_dim, mem_dim)

        self.device = device


    def forward(self, vid_feats, qns, qns_lengths, ans, ans_lengths, teacher_force_ratio=0.5, mode='train'):

        vid_outputs, vid_hidden = self.vid_encoder(vid_feats)
        qns_outputs, qns_hidden = self.qns_encoder(qns, qns_lengths)
        qns_outputs = qns_outputs.permute(1, 0, 2)
        # print(vid_outputs.size(), vid_hidden[0].size(), vid_hidden[1].size())
        # print(qns_outputs.size(), qns_hidden[0].size(), qns_hidden[1].size())
        """
        torch.Size([3, 128, 1024]) torch.Size([2, 3, 512]) torch.Size([2, 3, 512])
        torch.Size([3, 16, 1024]) torch.Size([2, 3, 512]) torch.Size([2, 3, 512])
        """

        word_num, batch_size, feat_dim = qns_outputs.size()
        r = torch.zeros((batch_size, vid_hidden[0].shape[-1])).to(self.device)

        for word in qns_outputs:
            r, beta_r = self.att_q2v(word, vid_outputs, r)

        vid_outputs = vid_outputs.permute(1, 0, 2) # change to fnum, batch_size, feat_dim
        qns_outputs = qns_outputs.permute(1, 0, 2) # change to batch_size, word_num, feat_dim
        w = torch.zeros((batch_size, vid_hidden[0].shape[-1])).to(self.device)
        for frame in vid_outputs:
            w, beta_w = self.att_v2q(frame, qns_outputs, w)

        hidden = (torch.cat((r.unsqueeze(0), w.unsqueeze(0)), dim=0),
                  torch.cat((vid_hidden[1][0].unsqueeze(0), qns_hidden[1][0].unsqueeze(0)), dim=0))

        if mode == 'train':
            vocab_size = self.ans_decoder.vocab_size
            batch_size = ans.shape[0]
            ans_len = ans.shape[1]
            input = ans[:, 0]
            outputs = torch.zeros(batch_size, ans_len, vocab_size).to(self.device)
            for t in range(0, ans_len):
                output, hidden = self.ans_decoder(hidden, input)
                outputs[:, t] = output
                teacher_force = rd.random() < teacher_force_ratio
                top1 = output.argmax(1)
                input = ans[:, t] if teacher_force else top1
        else:
            start = torch.LongTensor([1] * batch_size).to(self.device)
            outputs = self.ans_decoder.sample(hidden, start)
        return outputs
