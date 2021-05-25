import torch
import torch.nn as nn
import random as rd

class EVQA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, ans_decoder, device):
        """
        :param vid_encoder:
        :param qns_encoder:
        :param ans_decoder:
        :param device:
        """
        super(EVQA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.ans_decoder = ans_decoder
        self.device = device

    def forward(self, vid_feats, qns, qns_lengths, ans, ans_lengths, teacher_force_ratio=0.5, mode='train'):

        vid_outputs, vid_hidden = self.vid_encoder(vid_feats)
        qns_outputs, qns_hidden = self.qns_encoder(qns, qns_lengths)


        hidden = qns_hidden[0] +vid_hidden[0]
        batch_size = qns.shape[0]

        if mode == 'train':
            vocab_size = self.ans_decoder.vocab_size
            ans_len = ans.shape[1]
            input = ans[:, 0]
            outputs = torch.zeros(batch_size, ans_len, vocab_size).to(self.device)
            for t in range(0, ans_len):
                output, hidden = self.ans_decoder(qns_outputs, hidden, input)
                outputs[:,t] = output
                teacher_force = rd.random() < teacher_force_ratio
                top1 = output.argmax(1)
                input = ans[:, t] if teacher_force else top1
        else:
            start = torch.LongTensor([1] * batch_size).to(self.device)
            outputs = self.ans_decoder.sample(qns_outputs, hidden, start)

        return outputs
