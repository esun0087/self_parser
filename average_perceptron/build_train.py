"""
build train data
"""
import codecs
import jieba
import re
hanzi = re.compile(r"[\u4e00-\u9fa5]+")
def build_w_lable(word):
    ans = []
    if not hanzi.match(word):
        return []
    if len(word) == 1:
        ans.append( (word, 'o'))
    else:
        for i, n in enumerate(word):
            if i == 0:
                ans.append( (n, 'b'))
            elif i == len(word) - 1:
                ans.append( (n, 'e'))
            else:
                ans.append( (n, 'm'))
    return ans



def build_train_single_thread():
    with codecs.open("data/train.txt", "w", "utf-8") as f:
        i = 0
        for line in codecs.open("data/pao_mo_zhi_xia.txt", "r", "utf-8"):
            line = [i.strip() for i in line.split() if i.strip()]
            results = []
            for short_line in line:
                ans = []
                for w in jieba.cut(short_line.strip()):
                    a = build_w_lable(w)
                    if a:
                        ans.extend( a)
                ans.append((".", "."))
                results.extend(ans)
            for w,e in results:
                f.write ("\t".join((w,e)) + "\n")
            i +=1
            print (i)
            # break
import random
def build_train_for_pytorch():
    results = []
    i = 0
    for line in codecs.open("data/pao_mo_zhi_xia.txt", "r", "utf-8"):
        line = [i.strip() for i in line.split() if i.strip()]
        for short_line in line:
            if not hanzi.match(short_line):
                continue
            ans = []
            for w in jieba.cut(short_line.strip()):
                ans.append (w)
            results.append(ans)
        i +=1
        print (i)
            # break
    with codecs.open("data/train_pytorch.txt", "w", "utf-8") as f1:
        with codecs.open("data/dev_pytorch.txt", "w", "utf-8") as f2:
            with codecs.open("data/test_pytorch.txt", "w", "utf-8") as f3:
                for line in results:
                    a = random.randrange(1, 11)
                    if a <= 7:
                        f1.write("\t".join(line) + "\n")
                    elif a > 7 and a <= 9:
                        f2.write("\t".join(line) + "\n")
                    else:
                        f3.write("\t".join(line) + "\n")




if __name__ == '__main__':
    # multiprocessing.freeze_support()
    build_train_for_pytorch()
