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

vocb = set()
def get_n_gram( word, n):
    ret = [(w, i) for i, w in enumerate(word)]
    ans = [w for w in word]
    for i in range(1, n + 1):
        tmp = []
        for w, i in ret:
            if i + 1 < len(word):
                tmp.append((w + word[i + 1], i + 1))
                ans.append(w + word[i + 1])
        ret = tmp
    for i in ans:
        vocb.add(i)
    pinyins = lazy_pinyin(word)
    # pinyins = []
    return ans + pinyins


word_to_idx = pickle.load(open('word2idx', 'rb'))
idx_to_word = pickle.load(open('idx2word', 'rb'))
feat_to_idx = pickle.load(open('feat2idx', 'rb'))
idx_to_feat = pickle.load(open('idx2feat', 'rb'))
print ("len", len(word_to_idx))

class NgramModel(nn.Module):
    def __init__(self, fea_len,vocb_size):
        super(NgramModel, self).__init__()
        self.fea_len = fea_len
        self.vocb_size = vocb_size
        self.linear1 = nn.Linear(self.fea_len, 128)
        self.linear2 = nn.Linear(128, self.vocb_size)

    def forward(self, x):
        out = self.linear1(x.view(1,-1))
        out = F.tanh(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob

ngrammodel = ngrammodel = NgramModel(len(feat_to_idx), len(word_to_idx))
ngrammodel.load_state_dict(torch.load("ngrammodel.pt"))
pickle.dump(feat_to_idx,  open("feat2idx", 'wb'))
pickle.dump(idx_to_feat,  open("idx2feat", 'wb'))
torch.save(ngrammodel.state_dict(), "ngrammodel.pt")
word = "黑赛有默"
feat = [0 for i in range(len(feat_to_idx))]
for i in get_n_gram(word, len(word)):
    if i in feat_to_idx:
        feat[feat_to_idx[i]] += 1.0
feat = torch.tensor(feat)
out = ngrammodel(feat)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.item()]
print(' predict word is {}'.format(predict_word))