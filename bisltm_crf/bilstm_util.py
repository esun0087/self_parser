# 作者: Robert Guthrie

import torch
import torch.autograd as autograd
import codecs
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
torch.manual_seed(1)
def to_scalar(var):
    # 返回 python 浮点数 (float)
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # 以 python 整数的形式返回 argmax
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# 使用数值上稳定的方法为前向算法计算指数和的对数
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


m_len = 50

def get_word_index():
    w_idx = {".":0}
    train_f = r"D:\study\nlp\self_parser\average_perceptron\data\train.txt"
    for line in codecs.open(train_f, "r", "utf-8"):
        w = line.strip().split()[0]
        if w not in w_idx:
            w_idx[w] = len(w_idx)
    train_f = r"D:\study\nlp\self_parser\average_perceptron\data\test.txt"
    for line in codecs.open(train_f, "r", "utf-8"):
        w = line.strip().split()[0]
        if w not in w_idx:
            w_idx[w] = len(w_idx)
    return w_idx

def get_tag_index():
    tah_idx = {".":0, START_TAG:1, STOP_TAG:2}
    train_f = r"D:\study\nlp\self_parser\average_perceptron\data\train.txt"

    for line in codecs.open(train_f, "r", "utf-8"):
        w = line.strip().split()[1]
        if w not in tah_idx:
            tah_idx[w] = len(tah_idx)
    return tah_idx


def get_train_data():
    data = []
    juzi = []
    train_f = r"D:\study\nlp\self_parser\average_perceptron\data\train.txt"

    for line in codecs.open(train_f, "r","utf-8"):
        line = line.strip().split()
        if line[0] == '.':
            if juzi and len(juzi) <= m_len:
                data.append(([i[0] for i in juzi], [i[1] for i in juzi]))
            juzi = []
            continue
        juzi.append(line)

    return data

def get_test_data1():
    data = []
    juzi = []
    train_f = r"D:\study\nlp\self_parser\average_perceptron\data\test.txt"


    for line in codecs.open(train_f, "r","utf-8"):
        line = line.strip().split()
        if len(line) ==1:
            line = (line, 'x')
        if line[0] == '.':
            if juzi and len(juzi) <= m_len:
                data.append(([i[0] for i in juzi], [i[1] for i in juzi]))
            juzi = []
            continue
        juzi.append(line)

    return data

def get_test_data():
    data = []
    juzi = []
    train_f = r"D:\study\nlp\self_parser\average_perceptron\data\test_tmp.txt"

    for line in codecs.open(train_f, "r","utf-8"):
        line = line.strip().split()
        if len(line) ==1:
            line = (line, 'x')
        if line[0] == '.':
            if juzi and len(juzi) <= m_len:
                data.append(([i[0] for i in juzi], [i[1] for i in juzi]))
            juzi = []
            continue
        juzi.append(line)

    return data

def load_checkpoint(filename, model = None, g2c = False):
    print("loading model...")
    if g2c: # load weights into CPU
        checkpoint = torch.load(filename, map_location = lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])

def save_checkpoint(filename, model, epoch):
    if filename and model:
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        torch.save(checkpoint, filename + ".epoch")
        print("saved model at epoch %d" % epoch)

def prepare_sequence(seq, to_ix):
    if all([True if i in to_ix else False for i in seq ]):
        idxs = [to_ix[w] for w in seq]
        tensor = torch.LongTensor(idxs)
        return autograd.Variable(tensor)
    return None





