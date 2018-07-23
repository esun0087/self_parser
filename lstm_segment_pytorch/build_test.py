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
    with codecs.open(r"D:\study\nlp\self_parser\average_perceptron\data\test.txt", "w", "utf-8") as f:
        i = 0
        for line in codecs.open(r"D:\study\nlp\self_parser\average_perceptron\data\test_pytorch.txt", "r", "utf-8"):
            line = [i.strip() for i in line.split("，") if i.strip()]
            results = []
            for short_line in line:
                ans = []

                if short_line.strip():
                    for x in short_line.split():
                        a = build_w_lable(x)
                        if a:
                            ans.extend(a)
                ans.append((".", "."))
                results.extend(ans)
            for w,e in results:
                f.write ("\t".join((w,e)) + "\n")
            i +=1
            print (i)
            # break


if __name__ == '__main__':
    build_train_single_thread()