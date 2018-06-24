#coding:utf-8
from expr import *
import string
import codecs

symbols = ['[', ']', '<', '>', '|', '(', ')', ',']

class SemanticParser:
    def __init__(self):
        self.d = {}
    def lood_ahead(self,right_expr):
        token = ''
        for i in right_expr:
            if i.strip() == '':
                return token
            if i.strip() in symbols:
                if token.strip():
                    return token
                return i
            token += i
        return token

    def get_next_token(self,right_expr):
        token = ''
        block = 0
        for i, c in enumerate(right_expr):
            if c == '{':
                block += 1
                token += c
                continue
            if c == ',':
                if block > 0:
                    token += c
                    continue
            if c == '}':
                block -= 1

            if c not in symbols + list(string.whitespace):
                token += c
                continue

            if c.strip() == '':
                if token.strip() != "":
                    yield token, i
                    token = ''
            if c in symbols:
                if token.strip() != "":
                    yield token, i
                yield c, i
                token = ''
        if token.strip() != "":
            yield token, len(right_expr)


    def parse_or_list(self,op_stack, token_stack):
        opt_tmp = []
        while op_stack != [] and token_stack != [] and op_stack[-1][0] in ['|'] and op_stack[-1][1] < token_stack[-1][-1]:
            opt_tmp.append(token_stack.pop())
            op_stack.pop()

        if opt_tmp:
            opt_tmp.append(token_stack.pop())
            opt_tmp.reverse()
            index = opt_tmp[0][-1]
            opt_tmp = [k[0] for k in opt_tmp]
            token_stack.append((Or_List_Token(opt_tmp), index))


    def parse_expr_by_token(self,right_expr):
        op_stack = []
        token_stack = []
        key,right_expr = right_expr.split("=")
        key, right_expr = key.strip(), right_expr.strip()
        for token, last_index in self.get_next_token(right_expr):
            if token == '|':
                op_stack.append((token, last_index))
            elif token == '[':
                op_stack.append((token, last_index))
            elif token == '<':
                op_stack.append((token, last_index))
            elif token == 'magic':
                op_stack.append((token, last_index))
            elif token == '(':
                op_stack.append((token, last_index))
            elif token == ',':
                continue
            elif token == ']':
                self.parse_or_list(op_stack, token_stack)
                tmp = []
                while op_stack != [] and token_stack != [] and op_stack[-1][0] in ['[', '<'] and op_stack[-1][1] < \
                        token_stack[-1][-1]:
                    tmp.append(token_stack.pop())
                op_stack.pop()
                tmp.reverse()
                if tmp:
                    index = tmp[0][-1]
                    tmp = [k[0] for k in tmp]
                    token_stack.append((Opt_List_Token(tmp), index))
            elif token == '>':
                tmp = []
                while op_stack != [] and token_stack != [] and op_stack[-1][0] in ['<'] and op_stack[-1][1] < \
                        token_stack[-1][-1]:
                    tmp.append(token_stack.pop())
                op_stack.pop()
                tmp.reverse()
                if len(tmp) == 1:
                    index = tmp[0][-1]
                    w = Ref_Token(tmp[0][0], self.d)
                    token_stack.append((w, index))
            elif token == ')':
                tmp = []
                while op_stack != [] and token_stack != [] and op_stack[-1][0] in ['('] and op_stack[-1][1] < \
                        token_stack[-1][-1]:
                    tmp.append(token_stack.pop())
                op_stack.pop()
                tmp.reverse()
                if op_stack[-1][0] == 'magic':
                    if len(tmp) == 3:
                        index = tmp[0][-1]
                        w = MagicToken(tmp[1][0],  tmp[0][0], tmp[2][0])
                        token_stack.append((w, index))
                    op_stack.pop()
            else:
                if token.strip() != '':
                    token_stack.append((WorldToken(token), last_index))
                if self.lood_ahead(right_expr[last_index:]) != '|':
                    self.parse_or_list(op_stack, token_stack)

        self.parse_or_list(op_stack, token_stack)
        tmp = []
        while op_stack != [] and op_stack[-1] != '[' and token_stack != []:
            tmp.append(token_stack.pop())
        if tmp:
            index = tmp[0][-1]
            tmp = [k[0] for k in tmp]
            token_stack.append((Opt_List_Token(tmp), index))
        token_stack = [k[0] for k in token_stack]
        if len(token_stack) == 1:
            self.d[key] = token_stack[0]
        else:
            self.d[key] = List_Token(token_stack)
        return self.d[key]
    def parse(self, sent):
        root_token = self.d['root']
        return root_token.parse(sent)
    def parse_file(self, f_name):
        for line in codecs.open(f_name,'r', 'utf-8'):
            self.parse_expr_by_token(line.strip())
    def get_magic_token(self, parse_ans):
        return [p for p in parse_ans[0] if type(p) == MagicMatchResult]


if __name__ == '__main__':
    # parse_expr_by_token("[[a b] | c <f>] d e | x | p <zz> magic(x c v, aa)")
    get = SemanticParser()
    get.parse_expr_by_token("a=[magic([123 | 456],a,45)]")
