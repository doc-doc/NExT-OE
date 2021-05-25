import torch
import torch.nn as nn
import random as rd
import sys
sys.path.insert(0, 'networks')
from Attention import TempAttention
from memory_rand import MemoryRamTwoStreamModule, MemoryRamModule, MMModule


class HME(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, ans_decoder, max_len_v, max_len_q, device, input_drop_p=0.2):
        """
        Heterogeneous memory enhanced multimodal attention model for video question answering (CVPR19)

        """
        super(HME, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.ans_decoder = ans_decoder

        dim = qns_encoder.dim_hidden

        self.temp_att_a = TempAttention(dim * 2, dim * 2, hidden_dim=256)
        self.temp_att_m = TempAttention(dim * 2, dim * 2, hidden_dim=256)
        self.mrm_vid = MemoryRamTwoStreamModule(dim, dim, max_len_v, device)
        self.mrm_txt = MemoryRamModule(dim, dim, max_len_q, device)

        self.mm_module_v1 = MMModule(dim, input_drop_p, device)

        self.linear_vid = nn.Linear(dim*2, dim)
        self.linear_qns = nn.Linear(dim*2, dim)
        self.linear_mem = nn.Linear(dim*2, dim)
        self.vq2word_hme = nn.Linear(dim*3, dim*2)
        self._init_weights()
        self.device = device

    def _init_weights(self):
        """
        initialize the linear weights
        :return:
        """
        nn.init.xavier_normal_(self.linear_vid.weight)
        nn.init.xavier_normal_(self.linear_qns.weight)
        nn.init.xavier_normal_(self.linear_mem.weight)
        nn.init.xavier_normal_(self.vq2word_hme.weight)


    def forward(self, vid_feats, qns, qns_lengths, ans, ans_lengths, teacher_force_ratio=0.5, iter_num=3, mode='train'):
        """
        """

        outputs_app_l1, outputs_app_l2, outputs_motion_l1, outputs_motion_l2 = self.vid_encoder(vid_feats) #(batch_size, fnum, feat_dim)

        outputs_app = torch.cat((outputs_app_l1, outputs_app_l2), dim=-1)
        outputs_motion = torch.cat((outputs_motion_l1, outputs_motion_l2), dim=-1)

        batch_size, fnum, vid_feat_dim = outputs_app.size()

        qns_output, qns_hidden = self.qns_encoder(qns, qns_lengths)
        # print(qns_output.shape, qns_hidden[0].shape) #torch.Size([10, 23, 256]) torch.Size([2, 10, 256])

        # qns_output = qns_output.permute(1, 0, 2)
        batch_size, seq_len, qns_feat_dim = qns_output.size()

        qns_embed = qns_hidden[0].permute(1, 0, 2).contiguous().view(batch_size, -1) #(batch_size, feat_dim)

        # Apply temporal attention
        att_app, beta_app = self.temp_att_a(qns_embed, outputs_app)
        att_motion, beta_motion = self.temp_att_m(qns_embed, outputs_motion)
        tmp_app_motion = torch.cat((outputs_app_l2[:, -1, :], outputs_motion_l2[:, -1, :]), dim=-1)

        mem_output = torch.zeros(batch_size, vid_feat_dim).to(self.device)

        for bs in range(batch_size):
            mem_ram_vid = self.mrm_vid(outputs_app_l2[bs], outputs_motion_l2[bs], fnum)
            cur_qns = qns_output[bs][:qns_lengths[bs]]
            mem_ram_txt = self.mrm_txt(cur_qns, qns_lengths[bs]) #should remove padded zeros
            mem_output[bs] = self.mm_module_v1(tmp_app_motion[bs].unsqueeze(0), mem_ram_vid, mem_ram_txt, iter_num)
            """
            (64, 256) (22, 256) (1, 512)
            """
        app_trans = torch.tanh(self.linear_vid(att_app))
        motion_trans = torch.tanh(self.linear_vid(att_motion))
        mem_trans = torch.tanh(self.linear_mem(mem_output))

        encoder_outputs = torch.cat((app_trans, motion_trans, mem_trans), dim=1)
        decoder_inputs = self.vq2word_hme(encoder_outputs)
        hidden = qns_hidden
        if mode == 'train':
            vocab_size = self.ans_decoder.vocab_size
            ans_len = ans.shape[1]
            input = ans[:, 0]

            outputs = torch.zeros(batch_size, ans_len, vocab_size).to(self.device)

            for t in range(0, ans_len):
                output, hidden = self.ans_decoder(decoder_inputs, hidden, input)
                outputs[:, t] = output
                teacher_force = rd.random() < teacher_force_ratio
                top1 = output.argmax(1)
                input = ans[:, t] if teacher_force else top1
        else:
            start = torch.LongTensor([1] * batch_size).to(self.device)
            outputs = self.ans_decoder.sample(decoder_inputs, hidden, start)

        return outputs