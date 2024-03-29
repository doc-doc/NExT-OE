from utils import *
from metrics import *
import pandas as pd
from pywsd.utils import lemmatize_sentence

#use stopwords tailored for NExT-QA
stopwords = load_file('stopwords.txt')
def remove_stop(sentence):

    words = lemmatize_sentence(sentence)
    words = [w for w in words if not w in stopwords]
    return ' '.join(words)


def evaluate(res_file, ref_file, ref_file_add):
    """
    :param res_file:
    :param ref_file:
    :return:
    """
    res = load_file(res_file)

    multi_ref_ans = False
    if osp.exists(ref_file_add):
        add_ref = load_file(ref_file_add)
        multi_ref_ans = True 
    refer = pd.read_csv(ref_file)
    ref_num = len(refer)
    group_dict = {'CW': [], 'CH': [], 'TN': [], 'TC': [], 'DL': [], 'DB':[], 'DC': [], 'DO': []}
    for idx in range(ref_num):
        sample = refer.loc[idx]
        qtype = sample['type']
        if qtype in ['TN', 'TP']: qtype = 'TN'
        group_dict[qtype].append(idx)
    wups0 = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DB': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    wups9 = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DB': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    wups0_e, wups0_t, wups0_c = 0, 0, 0
    wups0_all, wups9_all = 0, 0

    num = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0, 'DB': 0, 'DC': 0, 'DL': 0, 'DO': 0}
    over_num = {'C':0, 'T':0, 'D':0}
    ref_num = 0
    for qtype, ids in group_dict.items():
        for id in ids:
            sample = refer.loc[id]
            video, qid, ans, qns = str(sample['video']), str(sample['qid']), str(sample['answer']), str(sample['question'])
            num[qtype] += 1
            over_num[qtype[0]] += 1
            ref_num += 1

            pred_ans_src = res[video][qid]

            gt_ans = remove_stop(ans)
            pred_ans = remove_stop(pred_ans_src)
            if multi_ref_ans and (video in add_ref):
                gt_ans_add = remove_stop(add_ref[video][qid])
                if qtype == 'DC' or qtype == 'DB':
                    cur_0 = 1 if pred_ans == gt_ans_add or pred_ans == gt_ans else 0
                    cur_9 = cur_0
                else:
                    cur_0 = max(get_wups(pred_ans, gt_ans, 0), get_wups(pred_ans, gt_ans_add, 0))
                    cur_9 = max(get_wups(pred_ans, gt_ans, 0.9), get_wups(pred_ans, gt_ans_add, 0.9))
            else:
                if qtype == 'DC' or qtype == 'DB':
                    cur_0 = 1 if pred_ans == gt_ans else 0
                    cur_9 = cur_0
                else:
                    cur_0 = get_wups(pred_ans,  gt_ans, 0)
                    cur_9 = get_wups(pred_ans, gt_ans, 0.9)
            wups0[qtype] += cur_0
            wups9[qtype] += cur_9


        wups0_all += wups0[qtype]
        wups9_all += wups9[qtype]
        if qtype[0] == 'C':
            wups0_e += wups0[qtype]
        if qtype[0] == 'T':
            wups0_t += wups0[qtype]
        if qtype[0] == 'D':
            wups0_c += wups0[qtype]

        wups0[qtype] = wups0[qtype]/num[qtype]
        wups9[qtype] = wups9[qtype]/num[qtype]

    num_e = over_num['C']
    num_t =  over_num['T']
    num_c = over_num['D']

    wups0_e /= num_e
    wups0_t /= num_t
    wups0_c /= num_c

    wups0_all /= ref_num
    wups9_all /= ref_num

    for k in wups0:
        wups0[k] = wups0[k] * 100
        wups9[k] = wups9[k] * 100

    wups0_e *= 100
    wups0_t *= 100
    wups0_c *= 100
    wups0_all *= 100

    print('CW\tCH\tWUPS_C\tTPN\tTC\tWUPS_T\tDB\tDC\tDL\tDO\tWUPS_D\tWUPS')
    print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'
          .format(wups0['CW'], wups0['CH'], wups0_e,  wups0['TN'], wups0['TC'],wups0_t,
                  wups0['DB'],wups0['DC'], wups0['DL'], wups0['DO'], wups0_c, wups0_all))



def main(filename, mode):
    res_dir = 'results'
    res_file = osp.join(res_dir, filename)
    print(f'Evaluate on {res_file}')
    ref_file = 'dataset/nextqa/{}.csv'.format(mode)
    ref_file_add = 'dataset/nextqa/add_reference_answer_{}.json'.format(mode)
    evaluate(res_file, ref_file, ref_file_add)


if __name__ == "__main__":

    mode = 'val'
    model = 'HGA'
    result_file = '{}-same-att-qns23ans7-{}-example.json'.format(model, mode)
    main(result_file, mode)
