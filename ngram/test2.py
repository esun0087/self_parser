#coding:utf-8
import re
import codecs
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pypinyin import pinyin, lazy_pinyin, Style

hanzi = re.compile(r"[\u4e00-\u9fa5]+")
test_sentence = [i.strip() for i in codecs.open("song_name.txt", "r", "utf-8").readlines() if hanzi.match(i.strip())]

def get_n_feat(feat_list, n):
    ret = [(w, i) for i, w in enumerate(feat_list)]
    ans = [w for w in feat_list]
    for j in range(1, n + 1):
        tmp = []
        for w, i in ret:
            if i + 1 < len(feat_list):
                tmp.append((w + feat_list[i + 1], i + 1))
                ans.append(w + feat_list[i + 1])
        ret = tmp
    return ans
def get_n_gram_feat( word, n):
    ans = get_n_feat(word, n)
    return ans

def get_pinyin_feat(word):
    py = lazy_pinyin(word)
    ans = get_n_feat(py, len(py))
    return ans


word_to_idx = pickle.load(open('word2idx_new', 'rb'))
idx_to_word = pickle.load(open('idx2word_new', 'rb'))
ngram_feat_to_idx = pickle.load(open('ngramfeat2idx', 'rb'))
ngram_idx_to_feat = pickle.load(open('ngramidx2feat', 'rb'))
pinyin_feat_to_idx = pickle.load(open('pinyinfeat2idx', 'rb'))
pinyin_idx_to_feat = pickle.load(open('pinyinidx2feat', 'rb'))
print ("len", len(word_to_idx))

class NgramModel(nn.Module):
    def __init__(self, ngram_fea_len,pinyin_fea_len,vocb_size):
        super(NgramModel, self).__init__()
        self.ngram_fea_len = ngram_fea_len
        self.pinyin_fea_len = pinyin_fea_len
        self.vocb_size = vocb_size
        self.linear1 = nn.Linear(self.ngram_fea_len, 128)
        self.linear2 = nn.Linear(self.pinyin_fea_len, 128)
        self.linear3 = nn.Linear(256, self.vocb_size)

    def forward(self, ngram_feat, pinyin_feat):
        ngram_out = self.linear1(ngram_feat)
        pinyin_out = self.linear2(pinyin_feat)

        out = torch.cat((ngram_out, pinyin_out))
        out = self.linear3(out)
        log_prob = F.log_softmax(out)
        return log_prob.unsqueeze(0)

ngrammodel = NgramModel(len(ngram_feat_to_idx), len(pinyin_feat_to_idx), len(word_to_idx))
ngrammodel.load_state_dict(torch.load("ngrammodel_new.pt"))
def predict(word):
    with torch.no_grad():
        ngram_feat  = [0.0 for i in range(len(ngram_feat_to_idx))]
        pinyin_feat  = [0.0 for i in range(len(pinyin_feat_to_idx))]

        for i in get_n_gram_feat(word, len(word)):
            if i in ngram_feat_to_idx:
                ngram_feat[ngram_feat_to_idx[i]] += 1.0
        for i in get_pinyin_feat(word):

            if i in pinyin_feat_to_idx:
                pinyin_feat[pinyin_feat_to_idx[i]] += 1.0

        ngram_feat, pinyin_feat = torch.tensor(ngram_feat), torch.tensor(pinyin_feat)
        out = ngrammodel(ngram_feat, pinyin_feat)
        _, predict_label = torch.max(out, 1)
        predict_word = idx_to_word[predict_label.item()]
        print(' predict word is {}'.format(predict_word))

predict("菊华台")
# predict("让心二圈起你")
# predict("大男人晓女孩")
# predict("决战前")
predict("许文强")