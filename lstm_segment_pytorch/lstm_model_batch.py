import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 5
class LSTMTaggerBatch(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTaggerBatch, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, sentence):

        embeds = self.word_embeddings(sentence)
        # print ("sentence shape", sentence.shape)
        # print("embeds shape ", embeds.shape, embeds.view(len(sentence), len(sentence[0]), -1).shape)
        embeds = embeds.view(len(sentence), len(sentence[0]), -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # print("lstmout ", lstm_out.shape)

        tag_space = self.hidden2tag(lstm_out)
        # print("tag_space shape", tag_space.shape)

        tag_scores = F.log_softmax(tag_space, dim=2)
        # print("tag_scores shape", tag_scores.shape)

        return tag_scores