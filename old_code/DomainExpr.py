#coding:utf-8
import codecs
from old_code.parse_expr import Expr

class DomainExpr(object):
    """
    对一个语法文件的解析,
    包括一个语法文件的所有语法
    """
    def __init__(self):
        self.all_token = {}
        self.root_expr = []

    def parse_file(self, file_in):
        for line in codecs.open(file_in):
            expr = Expr(line.strip(),d = self.all_token)
            expr.parse()
            if expr.key_word == None:
                continue
            if expr.key_word.startswith("<root"):
                self.root_expr.append(expr)
        self.flush()
    def eval(self, input_str):
        ans = []
        for expr in self.root_expr:
            ret = expr.eval(input_str)
            if ret:
                ans.append(ret)
        return ans


    def flush(self):
        tmp = self.all_token
        for k in tmp:
            for i in range(len(tmp[k].list)):
                if tmp[k].list[i][1] == None:
                    if tmp[k].list[i][0] in tmp:
                        tmp[k].list[i][1] = tmp[tmp[k].list[i][0]]
                    else:
                        print('%s key error' % tmp[k].list[0])