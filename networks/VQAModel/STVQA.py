import torch
import torch.nn as nn
import random as rd
import sys
sys.path.insert(0, 'networks')
from Attention import TempAttention, SpatialAttention


class STVQA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, ans_decoder, att_dim, device):
        """
        TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering (CVPR17)
        """
        super(STVQA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.ans_decoder = ans_decoder
        self.att_dim = att_dim

        self.spatial_att = SpatialAttention(qns_encoder.dim_hidden*2, vid_encoder.input_dim, hidden_dim=self.att_dim)
        self.temp_att = TempAttention(qns_encoder.dim_hidden*2, vid_encoder.dim_hidden*2, hidden_dim=self.att_dim)
        self.device = device
        self.FC = nn.Linear(att_dim*2, att_dim)


    def forward(self, vid_feats, qns, qns_lengths, ans, ans_lengths, teacher_force_ratio=0.5, mode='train'):
        """
        """
        qns_output_1, qns_hidden_1 = self.qns_encoder(qns, qns_lengths)
        n_layers, batch_size, qns_dim = qns_hidden_1[0].size()

        # Concatenate the dual-layer hidden as qns embedding
        qns_embed = qns_hidden_1[0].permute(1, 0, 2) # batch first
        qns_embed = qns_embed.reshape(batch_size, -1) #(batch_size, feat_dim*2)
        batch_size, fnum, vid_dim, w, h = vid_feats.size()

        # Apply spatial attention
        vid_att_feats = torch.zeros(batch_size, fnum, vid_dim).to(self.device)
        for bs in range(batch_size):
            vid_att_feats[bs], alpha = self.spatial_att(qns_embed[bs], vid_feats[bs])

        vid_outputs, vid_hidden = self.vid_encoder(vid_att_feats)

        qns_outputs, qns_hidden = self.qns_encoder(qns, qns_lengths, vid_hidden)

        """
        torch.Size([3, 128, 1024]) torch.Size([2, 3, 512]) torch.Size([2, 3, 512])
        torch.Size([16, 3, 1024]) torch.Size([2, 3, 512]) torch.Size([2, 3, 512])
        """
        qns_embed = qns_hidden[0].permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)

        # Apply temporal attention
        temp_att_outputs, beta = self.temp_att(qns_embed, vid_outputs)
        encoder_outputs = self.FC(qns_embed + temp_att_outputs)
        # hidden = qns_hidden
        hidden = encoder_outputs.unsqueeze(0)
        # print(hidden.size())

        if mode == 'train':
            vocab_size = self.ans_decoder.vocab_size
            ans_len = ans.shape[1]
            input = ans[:, 0]
            outputs = torch.zeros(batch_size, ans_len, vocab_size).to(self.device)
            for t in range(0, ans_len):
                # output, hidden = self.ans_decoder(encoder_outputs, hidden, input)
                output, hidden = self.ans_decoder(qns_outputs, hidden, input)
                outputs[:, t] = output
                teacher_force = rd.random() < teacher_force_ratio
                top1 = output.argmax(1)
                input = ans[:, t] if teacher_force else top1
        else:
            start = torch.LongTensor([1] * batch_size).to(self.device)
            # outputs = self.ans_decoder.sample(encoder_outputs, hidden, start)
            outputs = self.ans_decoder.sample(qns_outputs, hidden, start)

        return outputs
