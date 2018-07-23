
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from bilstm_util import *

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # 将LSTM的输出映射到标记空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 过渡参数矩阵. 条目 i,j 是
        # *从* j *到* i 的过渡的分数
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 这两句声明强制约束了我们不能
        # 向开始标记标注传递和从结束标注传递
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats):
        # 执行前向算法来计算分割函数
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG 包含所有的分数
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 将其包在一个变量类型中继而得到自动的反向传播
        forward_var = autograd.Variable(init_alphas)

        # 在句子中迭代
        for feat in feats:
            alphas_t = []  # 在这个时间步的前向变量
            for next_tag in range(self.tagset_size):
                # 对 emission 得分执行广播机制: 它总是相同的,
                # 不论前一个标注如何
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # trans_score 第 i 个条目是
                # 从i过渡到 next_tag 的分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # next_tag_var 第 i 个条目是在我们执行 对数-求和-指数 前
                # 边缘的值 (i -> next_tag)
                next_tag_var = forward_var + trans_score + emit_score
                # 这个标注的前向变量是
                # 对所有的分数执行 对数-求和-指数
                alphas_t.append(log_sum_exp(next_tag_var).unsqueeze(0))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # 给出标记序列的分数
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # 在对数空间中初始化维特比变量
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        # 在第 i 步的 forward_var 存放第 i-1 步的维特比变量
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # 存放这一步的后指针
            viterbivars_t = []  # 存放这一步的维特比变量
            for next_tag in range(self.tagset_size):
                # next_tag_var[i] 存放先前一步标注i的
                # 维特比变量, 加上了从标注 i 到 next_tag 的过渡
                # 的分数
                # 我们在这里并没有将 emission 分数包含进来, 因为
                # 最大值并不依赖于它们(我们在下面对它们进行的是相加)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].unsqueeze(0))
            # 现在将所有 emission 得分相加, 将 forward_var
            # 赋值到我们刚刚计算出来的维特比变量集合

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 过渡到 STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        # 跟着后指针去解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始的标签 (我们并不希望把这个返回到调用函数)
        start = best_path.pop()

        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # 不要把这和上面的 _forward_alg 混淆
        # 得到 BiLSTM 输出分数
        lstm_feats = self._get_lstm_features(sentence)

        # 给定特征, 找到最好的路径
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq