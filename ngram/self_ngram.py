#coding:utf-8
from pypinyin import pinyin, lazy_pinyin, Style
from collections import defaultdict
import codecs
import re
import pickle

vocb = set()
weights = {}
classes = set()
hanzi = re.compile(r"[\u4e00-\u9fa5]+")
ii = 0
tstamps = defaultdict(int)
totals = defaultdict(int)
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

def gen_feat(word):
    ngram_feat = get_n_gram_feat(word, len(word))
    pinyin_feat = get_pinyin_feat(word)
    feats = defaultdict(int)
    for f in ngram_feat + pinyin_feat:
        feats[f] += 1
    return feats

def predict(feats):
    scores = defaultdict(float)
    for f,v in feats.items():
        if f not in weights or v == 0:
            continue
        feats_weight = weights[f]
        for label, weight in feats_weight.items():
            scores[label] = weight * v
    return max(classes, key=lambda label: (scores[label], label))


def update(truth, guess, features):
    '''Update the feature weights.'''
    def upd_feat(c, f, w, v):
        param = (f, c)
        totals[param] += (ii - tstamps[param]) * w
        tstamps[param] = ii
        weights[f][c] = w + v

    global  ii
    ii += 1
    if truth == guess:
        return None
    for f in features:
        f_weight = weights.setdefault(f, {})
        upd_feat(truth, f, f_weight.get(truth, 0.0), 1.0)
        upd_feat(guess, f, f_weight.get(guess, 0.0), -1.0)
    return None
def make_tagdict( sentences):
    '''Make a tag dictionary for single-tag words.'''
    for word in sentences:
        classes.add(word)

def average_weights():
    '''Average weights from all iterations.'''
    for feat, feat_weight in weights.items():
        new_feat_weights = {}
        for clas, weight in feat_weight.items():
            param = (feat, clas)
            total = totals[param]
            total += (ii - tstamps[param]) * weight
            averaged = round(total / float(ii), 3)
            if averaged:
                new_feat_weights[clas] = averaged
        weights[feat] = new_feat_weights
    return None

def train_sent( sentences, save_loc=None, nr_iter=5):
    '''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
    controls the number of Perceptron training iterations.
    :param sentences: A list of (words, tags) tuples.
    :param save_loc: If not ``None``, saves a pickled model in this location.
    :param nr_iter: Number of training iterations.
    '''

    for iter_ in range(nr_iter):
        print(iter_)
        for words, tag in sentences:
            feats = gen_feat(words)
            guess = predict(feats)
            update(tag, guess, feats)
    average_weights()
    # Pickle as a binary file
    if save_loc is not None:
        pickle.dump((weights, classes),
                    open(save_loc, 'wb'), -1)
    return None

def tag(word):
    '''Tags a string `corpus`.'''
    features = gen_feat(word)
    tag1 = predict(features)
    print (tag1)
def train():
    training_data = []
    test_sentence = [i.strip() for i in codecs.open("new_song_name.txt", "r", "utf-8").readlines() if
                     hanzi.match(i.strip())]
    make_tagdict(test_sentence)
    for word in test_sentence:
        training_data.append((word, word))
    train_sent(training_data, save_loc="ngram_self", nr_iter=200)

def test():
    global weights, classes
    weights, classes = pickle.load(open("ngram_self", 'rb'))
    tag("行方")

test()

