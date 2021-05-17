from videoqa import *
import dataloader
from build_vocab import Vocabulary
from utils import *
import argparse
import eval_oe


def main(args):

    mode = args.mode
    if mode == 'train':
        batch_size = 64
        num_worker = 8
    else:
        batch_size = 64
        num_worker = 8
    spatial = False
    if spatial:
        #for STVQA only
        video_feature_path = '../data/feats/spatial/'
        video_feature_cache = '../data/feats/cache_spatial/'
    else:
        video_feature_path = '../data/feats/'
        video_feature_cache = '../data/feats/cache/'

    dataset = 'nextqa'
    sample_list_path = 'dataset/{}/'.format(dataset)
    #We separate the dicts for qns and ans, in case one wants to use different word-dicts for them.
    vocab_qns = pkload('dataset/{}/vocab.pkl'.format(dataset))
    vocab_ans = pkload('dataset/{}/vocab.pkl'.format(dataset))

    word_type = 'glove'
    glove_embed_qns = 'dataset/{}/{}_embed.npy'.format(dataset, word_type)
    glove_embed_ans = 'dataset/{}/{}_embed.npy'.format(dataset, word_type)
    checkpoint_path = 'models'
    model_type = 'HGA'

    model_prefix = 'same-att-qns23ans7'
    vis_step = 116
    lr_rate = 5e-5
    epoch_num = 100

    data_loader = dataloader.QALoader(batch_size, num_worker, video_feature_path, video_feature_cache,
                                      sample_list_path, vocab_qns, vocab_ans, True, False)

    train_loader, val_loader = data_loader.run(mode=mode)

    vqa = VideoQA(vocab_qns, vocab_ans, train_loader, val_loader, glove_embed_qns, glove_embed_ans,
                  checkpoint_path, model_type, model_prefix, vis_step,lr_rate, batch_size, epoch_num)

    ep = 36
    acc = 0.2163
    model_file = f'{model_type}-{model_prefix}-{ep}-{acc:.4f}.ckpt'

    if mode != 'train':
        result_file = f'{model_type}-{model_prefix}-{mode}.json'
        vqa.predict(model_file, result_file)
        eval_oe.main(result_file, mode)
    else:
        model_file = f'{model_type}-{model_prefix}-44-0.2140.ckpt'
        vqa.run(model_file, pre_trained=False)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', type=int,
                        default=0, help='gpu device id')
    parser.add_argument('--mode', dest='mode', type=str,
                        default='train', help='train or val')
    args = parser.parse_args()
    set_gpu_devices(args.gpu)
    main(args)
