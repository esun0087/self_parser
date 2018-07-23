import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from ngram_utils import *

# We will use Shakespeare Sonnet 2
train_data = get_train_data()
word_to_idx, idx_to_word = get_word_index(train_data)
print (len(word_to_idx))
trigram = get_tri_gram(train_data)


class NgramModel(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 10)
        self.linear2 = nn.Linear(10, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out, dim  = 1)
        return log_prob


ngrammodel = NgramModel(len(word_to_idx), CONTEXT_SIZE, 100)
criterion = nn.NLLLoss()
optimizer = optim.SGD(ngrammodel.parameters(), lr=1e-3)

for epoch in range(100):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    for data in trigram:
        word, label = data
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # forward
        out = ngrammodel(word)


        loss = criterion(out, label)
        running_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {:.6f}'.format(running_loss / len(word_to_idx)))

word, label = trigram[3]
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = ngrammodel(word)
print("out ", out.shape)
print("label ", label)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.data[0][0]]
print('real word is {}, predict word is {}'.format(label, predict_word))
