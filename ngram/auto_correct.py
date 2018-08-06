#coding:utf-8
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
import re
import codecs
import pickle
import random
from pypinyin import pinyin, lazy_pinyin, Style
CONTEXT_SIZE = 2
EMBEDDING_DIM = 20
# We will use Shakespeare Sonnet 2
hanzi = re.compile(r"[\u4e00-\u9fa5]+")
test_sentence = [i.strip() for i in codecs.open("new_song_name.txt", "r", "utf-8").readlines() if hanzi.match(i.strip())]

vocb = set()
pinyin_feats = set()
ngram_feats = set()


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
    vocb.add(word)
    return ans

def get_pinyin_feat(word):
    py = lazy_pinyin(word)
    ans = get_n_feat(py, len(py))
    return ans
def generate_train_feat():
    ans = []
    for word in test_sentence:
        n_grams_feat = get_n_gram_feat(word, len(word))
        pinyin_feat = get_pinyin_feat(word)
        ans.append(((tuple(n_grams_feat), tuple(pinyin_feat)), word))
    return ans
def add_feat(train_feat):
    for (a, b), c in train_feat:
        for i in a:
            ngram_feats.add(i)
        for i in b:
            pinyin_feats.add(i)

train_feat = generate_train_feat()
add_feat(train_feat)
ngram_feat_to_idx = {word: i for i, word in enumerate(ngram_feats)}
ngram_idx_to_feat = {ngram_feat_to_idx[word]: word for word in ngram_feat_to_idx}
pinyin_feat_to_idx = {word: i for i, word in enumerate(pinyin_feats)}
pinyin_idx_to_feat = {pinyin_feat_to_idx[word]: word for word in pinyin_feat_to_idx}
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
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


ngrammodel = NgramModel(len(ngram_feats), len(pinyin_feats), len(word_to_idx))
criterion = nn.NLLLoss()
optimizer = optim.Adam(ngrammodel.parameters(), lr=0.01)

def predict(word):
    with torch.no_grad():
        ngram_feat  = [0.0 for i in range(len(ngram_feats))]
        pinyin_feat  = [0.0 for i in range(len(pinyin_feats))]

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
for epoch in range(100):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    for data in train_feat:
        each_train_feat, label = data
        ngram_feat  = [0.0 for i in range(len(ngram_feats))]
        pinyin_feat  = [0.0 for i in range(len(pinyin_feats))]

        for i in each_train_feat[0]:
            if i in ngram_feats:
                ngram_feat[ngram_feat_to_idx[i]] += 1.0
        for i in each_train_feat[1]:
            if i in pinyin_feats:
                pinyin_feat[pinyin_feat_to_idx[i]] += 1.0
        ngram_feat, pinyin_feat = torch.tensor(ngram_feat), torch.tensor(pinyin_feat)
        out = ngrammodel(ngram_feat, pinyin_feat)
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # forward
        loss = criterion(out, label)
        running_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if running_loss / len(train_feat) < 0.00001:
        break
    predict("菊华台")
    predict("如果你是我的穿说")
    print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))
pickle.dump(word_to_idx,  open("word2idx_new", 'wb'))
pickle.dump(idx_to_word,  open("idx2word_new", 'wb'))
pickle.dump(ngram_feat_to_idx,  open("ngramfeat2idx", 'wb'))
pickle.dump(ngram_idx_to_feat,  open("ngramidx2feat", 'wb'))
pickle.dump(pinyin_feat_to_idx,  open("pinyinfeat2idx", 'wb'))
pickle.dump(pinyin_idx_to_feat,  open("pinyinidx2feat", 'wb'))
torch.save(ngrammodel.state_dict(), "ngrammodel_new.pt")

