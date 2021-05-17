from build_vocab import Vocabulary
from utils import *
import numpy as np
import random as rd
rd.seed(0)

def word2vec(vocab, glove_file, save_filename):
    glove = load_file(glove_file)
    word2vec = {}
    for i, line in enumerate(glove):
        if i == 0: continue  # for FastText
        line = line.split(' ')
        word2vec[line[0]] = np.array(line[1:]).astype(np.float)

    temp = []
    for word, vec in word2vec.items():
        temp.append(vec)
    temp = np.asarray(temp)
    print(temp.shape)
    row, col = temp.shape

    pad = np.mean(temp, axis=0)
    start = np.mean(temp[:int(row//2), :], axis=0)
    end = np.mean(temp[int(row//2):, :], axis=0)
    special_tokens = [pad, start, end]
    count = 0
    bad_words = []
    sort_idx_word = sorted(vocab.idx2word.items(), key=lambda k:k[0])
    glove_embed = np.zeros((len(vocab), 300))
    for row, item in enumerate(sort_idx_word):
        idx, word = item[0], item[1]
        if word in word2vec:
            glove_embed[row] = word2vec[word]
        else:
            if row < 3:
                glove_embed[row] = special_tokens[row]
            else:
                glove_embed[row] = np.random.randn(300)*0.4
            print(word)
            bad_words.append(word)
            count += 1
    print(glove_embed.shape)
    save_file(bad_words, 'bad_words_qns.json')
    np.save(save_filename, glove_embed)
    print(count)


def main():
    data_dir = 'dataset/nextqa/'
    vocab_file = osp.join(data_dir, 'vocab.pkl')
    vocab = pkload(vocab_file)
    glove_file = '../data/Vocabulary/glove.840B.300d.txt'
    save_filename = 'dataset/nextqa/glove_embed.npy'
    word2vec(vocab, glove_file, save_filename)

if __name__ == "__main__":
    main()
