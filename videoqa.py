from networks import EncoderRNN, DecoderRNN
from networks.VQAModel import EVQA, UATT, STVQA, CoMem, HME, HGA
from utils import *
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import time
from metrics import get_wups
from eval_oe import remove_stop

class VideoQA():
    def __init__(self, vocab_qns, vocab_ans, train_loader, val_loader, glove_embed_qns, glove_embed_ans,
                 checkpoint_path, model_type, model_prefix, vis_step,
                 lr_rate, batch_size, epoch_num):
        self.vocab_qns = vocab_qns
        self.vocab_ans = vocab_ans
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.glove_embed_qns = glove_embed_qns
        self.glove_embed_ans = glove_embed_ans
        self.model_dir = checkpoint_path
        self.model_type = model_type
        self.model_prefix = model_prefix
        self.vis_step = vis_step
        self.lr_rate = lr_rate
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None


    def build_model(self):

        vid_dim = 2048+2048
        hidden_dim = 512
        word_dim = 300
        qns_vocab_size = len(self.vocab_qns)
        ans_vocab_size = len(self.vocab_ans)
        max_ans_len = 7
        max_vid_len = 16
        max_qns_len = 23


        if self.model_type == 'EVQA' or self.model_type == 'BlindQA':
            #ICCV15, AAAI17
            vid_encoder = EncoderRNN.EncoderVid(vid_dim, hidden_dim, input_dropout_p=0.3, n_layers=1, rnn_dropout_p=0,
                                                bidirectional=False, rnn_cell='lstm')
            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, qns_vocab_size, self.glove_embed_qns, n_layers=1,
                                                input_dropout_p=0.3, rnn_dropout_p=0, bidirectional=False, rnn_cell='lstm')

            ans_decoder = DecoderRNN.AnsAttSeq(ans_vocab_size, max_ans_len, hidden_dim, word_dim, self.glove_embed_ans,
                                             n_layers=1, input_dropout_p=0.3, rnn_dropout_p=0, rnn_cell='gru')
            self.model = EVQA.EVQA(vid_encoder, qns_encoder, ans_decoder, self.device)

        elif self.model_type == 'UATT':
            #TIP17
            # hidden_dim = 512
            vid_encoder = EncoderRNN.EncoderVid(vid_dim, hidden_dim, input_dropout_p=0.3, bidirectional=True,
                                                rnn_cell='lstm')
            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, qns_vocab_size, self.glove_embed_qns,
                                                input_dropout_p=0.3, bidirectional=True, rnn_cell='lstm')

            ans_decoder = DecoderRNN.AnsUATT(ans_vocab_size, max_ans_len, hidden_dim, word_dim,
                                             self.glove_embed_ans, n_layers=2, input_dropout_p=0.3,
                                             rnn_dropout_p=0.5, rnn_cell='lstm')
            self.model = UATT.UATT(vid_encoder, qns_encoder, ans_decoder, self.device)

        elif self.model_type == 'STVQA':
            #CVPR17
            vid_dim = 2048 + 2048  # (64, 1024+2048, 7, 7)
            att_dim = 256
            hidden_dim = 256
            vid_encoder = EncoderRNN.EncoderVidSTVQA(vid_dim, hidden_dim, input_dropout_p=0.3, rnn_dropout_p=0,
                                                     n_layers=1, rnn_cell='lstm')
            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, qns_vocab_size, self.glove_embed_qns,
                                                input_dropout_p=0.3, rnn_dropout_p=0.5, n_layers=2, rnn_cell='lstm')
            ans_decoder = DecoderRNN.AnsAttSeq(ans_vocab_size, max_ans_len, hidden_dim, word_dim, self.glove_embed_ans,
                                              input_dropout_p=0.3, rnn_dropout_p=0, n_layers=1, rnn_cell='gru')
            self.model = STVQA.STVQA(vid_encoder, qns_encoder, ans_decoder, att_dim, self.device)


        elif self.model_type == 'CoMem':
            #CVPR18
            app_dim = 2048
            motion_dim = 2048
            hidden_dim = 256
            vid_encoder = EncoderRNN.EncoderVidCoMem(app_dim, motion_dim, hidden_dim, input_dropout_p=0.3,
                                                   bidirectional=False, rnn_cell='gru')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, qns_vocab_size, self.glove_embed_qns, n_layers=2,
                                                rnn_dropout_p=0.5, input_dropout_p=0.3, bidirectional=False, rnn_cell='gru')

            ans_decoder = DecoderRNN.AnsAttSeq(ans_vocab_size, max_ans_len, hidden_dim, word_dim, self.glove_embed_ans,
                                             n_layers=1, input_dropout_p=0.3, rnn_dropout_p=0, rnn_cell='gru')

            self.model = CoMem.CoMem(vid_encoder, qns_encoder, ans_decoder, max_vid_len, max_qns_len, self.device)


        elif self.model_type == 'HME':
            #CVPR19
            app_dim = 2048
            motion_dim = 2048
            vid_encoder = EncoderRNN.EncoderVidCoMem(app_dim, motion_dim, hidden_dim, input_dropout_p=0.3,
                                                   bidirectional=False, rnn_cell='lstm')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, qns_vocab_size, self.glove_embed_qns, n_layers=2,
                                                rnn_dropout_p=0.5, input_dropout_p=0.3, bidirectional=False, rnn_cell='lstm')

            ans_decoder = DecoderRNN.AnsHME(ans_vocab_size, max_ans_len, hidden_dim, word_dim, self.glove_embed_ans,
                                             n_layers=2, input_dropout_p=0.3, rnn_dropout_p=0.5, rnn_cell='lstm')

            self.model = HME.HME(vid_encoder, qns_encoder, ans_decoder, max_vid_len, max_qns_len, self.device)


        elif self.model_type == 'HGA':
            #AAAI20
            vid_encoder = EncoderRNN.EncoderVidHGA(vid_dim, hidden_dim, input_dropout_p=0.3,
                                                     bidirectional=False, rnn_cell='gru')

            qns_encoder = EncoderRNN.EncoderQnsHGA(word_dim, hidden_dim, qns_vocab_size, self.glove_embed_qns, n_layers=1,
                                                rnn_dropout_p=0, input_dropout_p=0.3, bidirectional=False,
                                                rnn_cell='gru')

            ans_decoder = DecoderRNN.AnsAttSeq(ans_vocab_size, max_ans_len, hidden_dim, word_dim, self.glove_embed_ans,
                                              n_layers=1, input_dropout_p=0.3, rnn_dropout_p=0, rnn_cell='gru')

            self.model = HGA.HGA(vid_encoder, qns_encoder, ans_decoder, max_vid_len, max_qns_len, self.device)


        params = [{'params':self.model.parameters()}]
        # params = [{'params': vid_encoder.parameters()}, {'params': qns_encoder.parameters()},
        #           {'params': ans_decoder.parameters(), 'lr': self.lr_rate}]
        self.optimizer = torch.optim.Adam(params = params, lr=self.lr_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, patience=5, verbose=True)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def save_model(self, epoch, loss):
        torch.save(self.model.state_dict(), osp.join(self.model_dir, '{}-{}-{}-{:.4f}.ckpt'
                                                     .format(self.model_type, self.model_prefix, epoch, loss)))

    def resume(self, model_file):
        """
        initialize model with pretrained weights
        :return:
        """
        model_path = osp.join(self.model_dir, model_file)
        print(f'Warm-starting from model {model_path}')
        model_dict = torch.load(model_path)
        new_model_dict = {}
        for k, v in self.model.state_dict().items():
            if k in model_dict:
                v = model_dict[k]

            new_model_dict[k] = v
        self.model.load_state_dict(new_model_dict)


    def run(self, model_file, pre_trained=False):
        self.build_model()
        best_eval_score = 0.0
        if pre_trained:
            self.resume(model_file)
            best_eval_score = self.eval(0)
            print('Initial Acc {:.4f}'.format(best_eval_score))

        for epoch in range(1, self.epoch_num):
            train_loss = self.train(epoch)
            eval_score = self.eval(epoch)
            print("==>Epoch:[{}/{}][Train Loss: {:.4f}  Val acc: {:.4f}]".
                  format(epoch, self.epoch_num, train_loss, eval_score))
            self.scheduler.step(eval_score)
            if eval_score > best_eval_score or pre_trained:
                best_eval_score = eval_score
                if epoch > 10 or pre_trained:
                    self.save_model(epoch, best_eval_score)


    def train(self, epoch):
        print('==>Epoch:[{}/{}][lr_rate: {}]'.format(epoch, self.epoch_num, self.optimizer.param_groups[0]['lr']))
        self.model.train()
        total_step = len(self.train_loader)
        epoch_loss = 0.0
        for iter, inputs in enumerate(self.train_loader):
            videos, targets_qns, qns_lengths, targets_ans, ans_lengths, video_names, qids, qtypes = inputs
            video_inputs = videos.to(self.device)
            qns_inputs = targets_qns.to(self.device)
            ans_inputs = targets_ans.to(self.device)
            prediction = self.model(video_inputs, qns_inputs, qns_lengths, ans_inputs, ans_lengths, 0.5)

            out_dim = prediction.shape[-1]
            prediction = prediction.view(-1, out_dim)
            ans_targets = ans_inputs.view(-1)

            loss = self.criterion(prediction, ans_targets)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if iter % self.vis_step == 0:
                print('\t[{}/{}]-{}-{:.4f}'.format(iter, total_step,cur_time, loss.item()))
            epoch_loss += loss.item()

        return epoch_loss / total_step


    def eval(self, epoch):
        print('==>Epoch:[{}/{}][validation stage]'.format(epoch, self.epoch_num))
        self.model.eval()
        total_step = len(self.val_loader)
        acc_count = 0
        with torch.no_grad():
            for iter, inputs in enumerate(self.val_loader):
                videos, targets_qns, qns_lengths, targets_ans, ans_lengths, video_names, qids, qtypes = inputs
                video_inputs = videos.to(self.device)
                qns_inputs = targets_qns.to(self.device)
                ans_inputs = targets_ans.to(self.device)
                prediction = self.model(video_inputs, qns_inputs, qns_lengths, ans_inputs, ans_lengths, mode='val')
                acc_count += get_acc_count(prediction, targets_ans, self.vocab_ans, qtypes)

        return acc_count*1.0 / ((total_step-1)*self.batch_size)


    def predict(self, model_file, res_file):
        """
        predict the answer with the trained model
        :param model_file:
        :return:
        """
        model_path = osp.join(self.model_dir, model_file)
        self.build_model()
        if self.model_type == 'HGA':
            self.resume(model_file)
        else:
            old_state_dict = torch.load(model_path)
            self.model.load_state_dict(old_state_dict)
        #self.resume()
        self.model.eval()
        total = len(self.val_loader)
        acc = 0
        results = {}
        with torch.no_grad():
            for iter, inputs in enumerate(self.val_loader):
                videos, targets_qns, qns_lengths, targets_ans, ans_lengths, video_names, qids, qtypes = inputs
                video_inputs = videos.to(self.device)
                qns_inputs = targets_qns.to(self.device)
                ans_inputs = targets_ans.to(self.device)
                # predict_ans_idxs = self.model.predict(video_inputs, qns_inputs, qns_lengths)
                predict_ans_idxs = self.model(video_inputs, qns_inputs, qns_lengths, ans_inputs, ans_lengths, mode='val')
                ans_idxs = predict_ans_idxs.cpu().numpy()
                targets_ans = targets_ans.numpy()
                targets_qns = targets_qns.numpy()
                for vname in video_names:
                    if vname not in results:
                        results[vname] = {}
                for bs, idx in enumerate(ans_idxs):
                    ans_pred = [self.vocab_ans.idx2word[ans_id] for ans_id in idx[1:] if ans_id >3] #the first 4 ids are reserved for special token
                    ans_pred = ' '.join(ans_pred)
                    groundtruth = [self.vocab_ans.idx2word[ans_id] for ans_id in targets_ans[bs][1:] if ans_id > 3]
                    groundtruth = ' '.join(groundtruth)
                    qns_text = [self.vocab_qns.idx2word[qns_id] for qns_id in targets_qns[bs][1:] if qns_id > 3]
                    # if qids[bs] not in results[video_names[bs]]:
                    qns_text = ' '.join(qns_text)
                    results[video_names[bs]][qids[bs]] = ans_pred
                    if ans_pred==groundtruth and ans_pred != '':
                        acc += 1
                    # print(f'[{iter}/{total}]{qns_text}? P:{ans_pred} G:{groundtruth}')

        save_file(results, f'results/{res_file}')


def get_acc_count(prediction, labels, vocab_ans, qtypes):
    """

    :param prediction:
    :param labels:
    :return:
    """
    preds = prediction.data.cpu().numpy()
    labels = np.asarray(labels)
    batch_size = labels.shape[0]
    score = 0
    for i in range(batch_size):
        pred = [j for j in preds[i] if j > 3]
        ans = [j for j in labels[i] if j > 3]
        pred_ans = ' '.join([vocab_ans.idx2word[id] for id in pred])
        gt_ans = ' '.join([vocab_ans.idx2word[id] for id in ans])
        pred_ans = remove_stop(pred_ans)
        gt_ans = remove_stop(gt_ans)
        cur_s = 0
        if qtypes[i] in ['CC', 'CB']:
            if gt_ans == pred_ans:
                cur_s = 1
        else:
            cur_s = get_wups(pred_ans, gt_ans, 0)
        score += cur_s


    return score








