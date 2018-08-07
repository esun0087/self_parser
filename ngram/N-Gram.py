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
feats = set()
def get_n_gram( word, n):
    ret = [(w, i) for i, w in enumerate(word)]
    ans = [w for w in word]
    for j in range(1, n + 1):
        tmp = []
        for w, i in ret:
            if i + 1 < len(word):
                tmp.append((w + word[i + 1], i + 1))
                ans.append(w + word[i + 1])
        ret = tmp
    vocb.add(word)
    pinyins = lazy_pinyin(word)
    return ans + pinyins

def get_tri():
    ans = []
    for word in test_sentence:
        n_grams = get_n_gram(word, len(word))
        ans.append((tuple(n_grams), word))
        for w in n_grams:
            feats.add(w)
    return ans
def add_pinyin():
    for word in test_sentence:
        for w in lazy_pinyin(word):
            feats.add(w)

trigram = get_tri()
add_pinyin()
feat_to_idx = {word: i for i, word in enumerate(feats)}
idx_to_feat = {feat_to_idx[word]: word for word in feat_to_idx}
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
print ("len", len(word_to_idx))

class NgramModel(nn.Module):
    def __init__(self, fea_len,vocb_size):
        super(NgramModel, self).__init__()
        self.fea_len = fea_len
        self.vocb_size = vocb_size
        self.linear1 = nn.Linear(self.fea_len, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, self.vocb_size)

    def forward(self, x):
        out = self.linear1(x.view(1,-1))
        out = self.linear3(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob


ngrammodel = NgramModel(len(feat_to_idx), len(word_to_idx))
criterion = nn.NLLLoss()
optimizer = optim.Adam(ngrammodel.parameters(), lr=0.01)

def predict(word):
    with torch.no_grad():
        feat = [0 for i in range(len(feat_to_idx))]
        for i in get_n_gram(word, len(word)):
            if i in feat_to_idx:
                feat[feat_to_idx[i]] += 1.0
        feat = torch.tensor(feat)
        out = ngrammodel(feat)
        _, predict_label = torch.max(out, 1)
        predict_word = idx_to_word[predict_label.item()]
        print(' predict word is {}'.format(predict_word))
for epoch in range(100):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    random.shuffle(trigram)
    for data in trigram:
        word, label = data
        feat = [0 for i in range(len(feat_to_idx))]
        for i in word:
            feat[feat_to_idx[i]] += 1.0
        feat = torch.tensor(feat)
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # forward
        out = ngrammodel(feat)
        loss = criterion(out, label)
        running_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if running_loss / len(trigram) < 0.00001:
        break
    predict("菊华台")
    predict("如果你是我的穿说")
    print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))
pickle.dump(word_to_idx,  open("word2idx", 'wb'))
pickle.dump(idx_to_word,  open("idx2word", 'wb'))
pickle.dump(feat_to_idx,  open("feat2idx", 'wb'))
pickle.dump(idx_to_feat,  open("idx2feat", 'wb'))
torch.save(ngrammodel.state_dict(), "ngrammodel.pt")

