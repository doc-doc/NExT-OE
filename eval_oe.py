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


def evaluate(res_file, ref_file):
    """
    :param res_file:
    :param ref_file:
    :return:
    """
    res = load_file(res_file)

    refer = pd.read_csv(ref_file)
    ref_num = len(refer)
    group_dict = {'EW': [], 'EH': [], 'TN': [], 'TC': [], 'CL': [], 'CB':[], 'CC': [], 'CO': []}
    for idx in range(ref_num):
        sample = refer.loc[idx]
        qtype = sample['type']
        if qtype in ['TN', 'TP']: qtype = 'TN'
        group_dict[qtype].append(idx)
    wups0 = {'EW': 0, 'EH': 0, 'TN': 0, 'TC': 0, 'CB': 0, 'CC': 0, 'CL': 0, 'CO': 0}
    wups9 = {'EW': 0, 'EH': 0, 'TN': 0, 'TC': 0, 'CB': 0, 'CC': 0, 'CL': 0, 'CO': 0}
    wups0_e, wups0_t, wups0_c = 0, 0, 0
    wups0_all, wups9_all = 0, 0

    num = {'EW': 0, 'EH': 0, 'TN': 0, 'TC': 0, 'CB': 0, 'CC': 0, 'CL': 0, 'CO': 0}
    over_num = {'E':0, 'T':0, 'C':0}
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

            if qtype == 'CC' or qtype == 'CB':
                cur_0 = 1 if pred_ans == gt_ans else 0
                cur_9 = cur_0
            else:
                cur_0 = get_wups(pred_ans,  gt_ans, 0)
                cur_9 = get_wups(pred_ans, gt_ans, 0.9)
            wups0[qtype] += cur_0
            wups9[qtype] += cur_9


        wups0_all += wups0[qtype]
        wups9_all += wups9[qtype]
        if qtype[0] == 'E':
            wups0_e += wups0[qtype]
        if qtype[0] == 'T':
            wups0_t += wups0[qtype]
        if qtype[0] == 'C':
            wups0_c += wups0[qtype]

        wups0[qtype] = wups0[qtype]/num[qtype]
        wups9[qtype] = wups9[qtype]/num[qtype]

    num_e = over_num['E']
    num_t =  over_num['T']
    num_c = over_num['C']

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

    print('EW\tEH\tAll\tTPN\tTC\tALL\tCB\tCC\tCL\tCO\tAll\tALL')
    print('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'
          .format(wups0['EW'], wups0['EH'], wups0_e,  wups0['TN'], wups0['TC'],wups0_t,
                  wups0['CB'],wups0['CC'], wups0['CL'], wups0['CO'], wups0_c, wups0_all))



def main(filename, mode):
    res_dir = 'results'
    res_file = osp.join(res_dir, filename)
    print(f'Evaluate on {res_file}')
    ref_file = 'dataset/nextqa/{}.csv'.format(mode)
    evaluate(res_file, ref_file)


if __name__ == "__main__":

    mode = 'val'
    model = 'HGA'
    result_file = '{}-same-att-qns23ans7-{}.json'.format(model, mode)
    main(result_file, mode)
