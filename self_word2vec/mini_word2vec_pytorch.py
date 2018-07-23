# encoding=utf-8
# Project: learn-pytorch
# Author: xingjunjie    github: @gavinxing
# Create Time: 29/07/2017 11:58 AM on PyCharm
# Basic template from http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from word2vec_utils import *
import numpy
from sklearn.metrics.pairwise import cosine_similarity

class CBOW(nn.Module):

    def __init__(self, embedding_size=100, vocab_size=None):
        super(CBOW, self).__init__()
        self.emb_dimension = embedding_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        embeds = embeds.view(1, -1)
        out = self.linear1(embeds)
        out = F.log_softmax(out, dim=1)

        return out
    def save_embedding(self, id2word, file_name):
        """Save all embeddings to file.

        As this class only record word id, so the map from id to word has to be transfered from outside.

        Args:
            id2word: map from word id to word.
            file_name: file name.
        Returns:
            None.
        """
        embedding = self.embeddings.weight.data.numpy()
        fout = codecs.open(file_name, 'w', "utf-8")
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
        fout.close()

    def get_emb(self, ids):
        return self.embeddings(ids)

# create your model and train.  here are some functions to help you make
# the data ready for use by your module

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def get_min_dis(line, embedding):
    ans = torch.mul(line, embedding)
    mod_line = torch.sqrt(torch.sum(torch.mul(line.type(torch.DoubleTensor),line.type(torch.DoubleTensor)), dim = 1))
    mod_emb = torch.sqrt(torch.sum(torch.mul(embedding.type(torch.DoubleTensor),embedding.type(torch.DoubleTensor)), dim = 1))

    ans = torch.sum(ans, dim = 1).type(torch.DoubleTensor)
    ans = [i / (mod_line[0] * j) for i, j in zip(ans, mod_emb)]

    ans = [(s, i) for i, s in enumerate(ans)]

    ans.sort(reverse=True)
    return ans[:10]

def test():
    f = codecs.open("embedding1.txt", "r", "utf-8")
    f.readline()
    all_embeddings = []
    all_words = []
    for i, line in enumerate(f):
        line = line.strip().split(' ')
        word = line[0]
        embedding = [float(x) for x in line[1:]]
        all_embeddings.append(embedding)
        all_words.append(word)
    all_embeddings = numpy.array(all_embeddings)
    words = ["羽毛球", "中国"]
    for ww in words:
        if ww in word_to_ix:
            wid = word_to_ix[ww]
            embedding = all_embeddings[wid:wid + 1]
            d = cosine_similarity(embedding, all_embeddings)[0]
            d = zip(all_words, d)
            d = sorted(d, key=lambda x: x[1], reverse=True)
            for w in d[:10]:
                print(w)
        print ("\n\n")


def test_for_predict(model):
    sent = "香港 羽毛球 不错".split()
    t = get_context(sent, CONTEXT_SIZE, 0)
    t_ids = torch.tensor([word_to_ix[i] for i in t if i in word_to_ix])
    model.zero_grad()
    ret = model(t_ids)
    ret = ret.squeeze()
    a = torch.topk(ret, 10)
    for score, index in zip(a[0], a[1]):
        print(sent[0], score, idx_to_word[index.item()])


if __name__ == '__main__':

    train_data = get_train_data()

    # By deriving a set from `raw_text`, we deduplicate the array

    word_to_ix,idx_to_word = get_word_index(train_data)
    vocab_size = len(word_to_ix)
    data = get_skip_gram(train_data)
    print ("tri gram data", len(data))
    with codecs.open("tri_gram.txt", "w", "utf-8") as f:
        for l in data:
            f.write("\t".join(l[0]) + "\t" + l[1] + "\n")


    loss_func = nn.NLLLoss()
    net = CBOW(embedding_size=EMBEDDING_SIZE, vocab_size=vocab_size)
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(30001):
        total_loss = 0
        for i, (context, target) in enumerate(data):
            context_var = make_context_vector(context, word_to_ix)
            net.zero_grad()
            log_probs = net(context_var)

            loss = loss_func(log_probs, torch.LongTensor([word_to_ix[target]]))

            loss.backward()
            optimizer.step()

            total_loss += loss.data
        if epoch % 1000 == 0:
            print("epoch", epoch, total_loss)
            net.save_embedding(idx_to_word, "embedding1.txt")
            test()
            # test_for_predict(net)