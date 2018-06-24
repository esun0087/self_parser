"""
ini config parse
[section]
key = value
coment ;
"""
import string
symbols = ['[', ']', ';', '=', ',']
class LineMeta:
    def __init__(self):
        self.key = None
        self.value = None
        self.comment = None
        self.type = None


def get_next_token(line):
    token = ''
    for i, c in enumerate(line):

        if c in string.ascii_letters + string.digits:
            token += c
            continue
        if c.strip() == '':
            if token.strip() != '':
                yield  token, i
            token = ''

        if c in symbols:
            if token.strip() != '':
                yield  token, i
            yield c, i
            token = ''
    if token.strip() != '':
        yield token, i

def parse_ini(line):
    token_stack = []
    op_stack = []
    meta = LineMeta()
    L = 0
    for token, last_index in get_next_token(line):
        L = last_index
        if token == '[':
            op_stack.append((token, last_index))
        elif token == ',':
            pass
        elif token == ';':
            if token_stack:
                meta.value = token_stack.pop()[0]
            break
        elif token == ']':
            tmp = []
            while op_stack != [] and token_stack != [] and op_stack[-1][0] in ['['] and op_stack[-1][1] < \
                    token_stack[-1][-1]:
                tmp.append(token_stack.pop())
            op_stack.pop()
            tmp.reverse()
            if meta.key:
                if not meta.value:
                    meta.value = tmp
            else:
                if len(tmp) == 1:
                    meta.key = tmp[0]
                    meta.type = 'section'
        elif token == '=':
            op_stack.append((token, last_index))
            tmp = token_stack.pop()
            meta.key = tmp[0]
            meta.type = 'varable_assign'
        else:
            token_stack.append((token, last_index))
    if L != len(line) - 1:
        meta.comment = line[L:]
    return meta



if __name__ == '__main__':
    # parse_ini('a=123;ssss')
    # parse_ini("[sec]")
    parse_ini('[a=[123,456,asf]];ssd')
a