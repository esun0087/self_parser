import codecs
import torch
from lstm_model import LSTMTagger
EMBEDDING_DIM, HIDDEN_DIM = 128, 128


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
    tah_idx = {".":0}
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
        juzi.append(line)
        if line[0] == '.':
            if juzi and len(juzi) <= m_len:
                data.append(([i[0] for i in juzi], [i[1] for i in juzi]))
            juzi = []
            continue
    return data

def get_test_data1():
    data = []
    juzi = []
    train_f = r"D:\study\nlp\self_parser\average_perceptron\data\test.txt"


    for line in codecs.open(train_f, "r","utf-8"):
        line = line.strip().split()
        if len(line) ==1:
            line = (line, 'x')
        juzi.append(line)
        if line[0] == '.':
            if juzi and len(juzi) <= m_len:
                data.append(([i[0] for i in juzi], [i[1] for i in juzi]))
            juzi = []
            continue
    return data

def get_test_data():
    data = []
    juzi = []
    train_f = r"D:\study\nlp\self_parser\average_perceptron\data\test_tmp.txt"


    for line in codecs.open(train_f, "r","utf-8"):
        line = line.strip().split()
        if len(line) ==1:
            line = (line, 'x')
        juzi.append(line)
        if line[0] == '.':
            if juzi and len(juzi) <= m_len:
                data.append(([i[0] for i in juzi], [i[1] for i in juzi]))
            juzi = []
            continue
    return data

def load_checkpoint(filename, model = None, g2c = False):
    print("loading model...")
    if g2c: # load weights into CPU
        checkpoint = torch.load(filename, map_location = lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])

def load_model():
    word_to_idx = get_word_index()
    tag_to_idx = get_tag_index()
    idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_idx))
    print(model)
    load_checkpoint("x.epoch", model)
    return model, word_to_idx, tag_to_idx, idx_to_tag

def save_checkpoint(filename, model, epoch):
    if filename and model:
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        torch.save(checkpoint, filename + ".epoch")
        print("saved model at epoch %d" % epoch)

def prepare_sequence(seq, to_ix):
    if all([True if i in to_ix else False for i in seq ]):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)
    return None


