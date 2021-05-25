import torch
import torch.nn as nn
import random as rd
from .Attention import TempAttentionHis, TempAttention, SpatialAttention
from .memory_rand import MemoryRamTwoStreamModule, MemoryRamModule, MMModule
from .memory_module import EpisodicMemory
from .q_v_transformer import CoAttention
from .gcn import AdjLearner, GCN
from block import fusions #pytorch >= 1.1.0


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

        # print(vid_outputs.size(), vid_hidden[0].size(), vid_hidden[1].size())
        # print(qns_outputs.size(), qns_hidden[0].size(), qns_hidden[1].size())
        """
        torch.Size([64, 128, 512]) torch.Size([1, 64 512]) torch.Size([1, 64, 512])
        torch.Size([16, 64, 512]) torch.Size([2, 64, 512]) torch.Size([2, 64, 512])
        """


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
            nn.Softmax(dim=-1))

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

