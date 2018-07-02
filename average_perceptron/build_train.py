"""
build train data
"""
import codecs
import jieba
import re
import multiprocessing
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

def prodess(line):
    line = [i.strip() for i in line.split() if i.strip()]
    result = []
    # for short_line in line:
    #     ans = []
    #     for w in jieba.cut(short_line.strip()):
    #         a = build_w_lable(w)
    #         if a:
    #             ans.extend(a)
    #         else:
    #             return []
    #     ans.append((".", "."))
    #     result.extend(ans)

    for short_line in line:
        ans = []
        a = build_w_lable(short_line)
        if a:
            ans.extend(a)
        else:
            return []
        ans.append((".", "."))
        result.extend(ans)
    return result

def multiprocess():
    with codecs.open("data/train_1.txt", "w", "utf-8") as f:
        p = multiprocessing.Pool(processes=8)
        i = 0
        results = []
        for line in codecs.open("D:\\data\\zh_data\wiki.zh.text.jian", "r", "utf-8"):
            result = p.apply_async(prodess, args=(line,))
            results.append(result)
            i += 1
            print (i)
        p.close()
        p.join()
        print ("ci over")
        i = 0
        for result in results:
            for w, e in result.get():
                f.write("\t".join((w, e)) + "\n")
            i += 1
            print("p " ,i)

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    build_train_single_thread()
