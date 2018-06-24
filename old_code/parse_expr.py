#coding:utf-8

"""
解析配置文件
"""
import re
from old_code.base_type import TokenType,EvalType
from copy import  deepcopy

class Expr(object):
    non_terminal = re.compile(r"<.*?>")
    terminal = re.compile(r"[^\s^<^>]+")
    blank = re.compile(r"[\s:]+")
    or_match = re.compile(r'[<>a-zA-Z]+(\s*\|\s*[<>a-zA-Z]+)+')
    optional_match = re.compile(r'\s*\[\s*[a-zA-Z]+(\s*[a-zA-Z]+)*\s*\]')
    magic_match = re.compile(r'magic\(\s*[\|a-zA-Z_]+\s*,\s*[a-zA-Z_]+\s*(,\s*[a-zA-Z_]+\s*){0,2}\)') # magic([b]<c>|,b,c)

    def __init__(self, s, key_word = None, token_type = None, d = None):
        self.s = s
        self.list = []
        self.key_word = key_word
        self.token_type = token_type
        self.d = d

    def get_global_token(self, token):
        return self.d[token] if token in self.d else None

    def add_global_token(self, token, expr):
        if token not in self.d:
            self.d[token] = expr
        if token in self.d and self.d[token] == None:
            self.d[token] = expr

    def add_token(self, token_info):
        self.list.append(token_info)

    def trime(self):
        blank_match = self.blank.match(self.s)
        if blank_match:
            self.s = self.s[blank_match.end():]


    def get_token(self):
        self.trime()
        or_match_res = self.or_match.match(self.s)
        option_match_res = self.optional_match.match(self.s)
        non_terminal_match = self.non_terminal.match(self.s)
        magic_match_res = self.magic_match.match(self.s)
        terminal_match = self.terminal.match(self.s)
        if or_match_res:
            self.s = self.s[or_match_res.end():]
            return or_match_res.group(), TokenType.OR_LIST
        if option_match_res:
            self.s = self.s[option_match_res.end():]
            return option_match_res.group(), TokenType.OPTION_LIST
        if non_terminal_match:
            self.s = self.s[non_terminal_match.end():]
            return non_terminal_match.group(), TokenType.NON_TERMINAL
        if magic_match_res:
            self.s = self.s[magic_match_res.end():]
            return magic_match_res.group(), TokenType.MAGIC
        if terminal_match:
            self.s = self.s[terminal_match.end():]
            return terminal_match.group(), TokenType.TERMINAL
        return None, TokenType.INVALID

    def parse_orlist(self, token_str):
        token_list = [i.strip() for i in token_str.split('|') if i.strip() != '']
        expr = Expr(token_str, '|'.join(token_list), TokenType.OR_LIST)
        for token in token_list:
            if self.non_terminal.match(token):
                self.add_global_token(token, self.get_global_token(token))
                expr.add_token([token, self.get_global_token(token)])
            elif self.terminal.match(token):
                self.add_global_token(token, Expr(token, token, TokenType.TERMINAL))
                expr.add_token([token, self.get_global_token(token)])

        self.add_token(['|'.join(token_list), expr])
        self.add_global_token('|'.join(token_list), expr)

    def parse_optionlist(self, token_str):
        """
        [a b c]
        :param token_str: 
        :return: 
        """
        token_str = token_str[1:-1]
        token_list = [i.strip() for i in token_str.split(' ') if i.strip() != '']
        expr = Expr(token_str, ' '.join(token_list), TokenType.OPTION_LIST)
        for token in token_list:
            if self.non_terminal.match(token):
                self.add_global_token(token, self.get_global_token(token))
                expr.add_token([token, self.get_global_token(token)])
            elif self.terminal.match(token):
                self.add_global_token(token, Expr(token, token, TokenType.TERMINAL))
                expr.add_token([token, self.get_global_token(token)])

        self.add_token((' '.join(token_list), expr))
        self.add_global_token(' '.join(token_list), expr)
    def be_comma(self):
        return self.s.startswith("#")
    def parse(self):
        self.trime()
        if self.be_comma():
            return

        while 1:
            token, t = self.get_token()
            if t == TokenType.TERMINAL:
                self.add_global_token(token, Expr(token, token, TokenType.TERMINAL))
                self.add_token([token, self.get_global_token(token)])
            elif t == TokenType.NON_TERMINAL:
                self.add_global_token(token, self.get_global_token(token))
                if self.key_word != None:
                    self.add_token([token, self.get_global_token(token)])
            elif t == TokenType.OR_LIST:
                self.parse_orlist(token)
            elif t == TokenType.OPTION_LIST:
                self.parse_optionlist(token)
            elif t == TokenType.MAGIC:
                self.parse_magic(token)
            else:
                break
            if self.key_word == None:
                self.key_word = token
                self.token_type = TokenType.NON_TERMINAL
        self.add_global_token(self.key_word, self)


    def parse_magic(self, token_str):
        """
        magic(x | y | z, tags, default, value)
        :return: 
        """
        def parse_normed_words(normed_words):
            ret_list = []
            for word in [i.strip() for i in normed_words.split("|")]:
                terminal_match = self.terminal.match(word)
                if not terminal_match:
                    continue
                word_expr = Expr(terminal_match.group(), terminal_match.group(), TokenType.TERMINAL)
                ret_list.append([terminal_match.group(), word_expr])
                self.add_global_token(terminal_match.group(), word_expr)
            return ret_list


        #split by ","
        token_str = token_str.strip()
        key_word = token_str.strip()
        key_word = key_word.replace(" ", "")
        token_list = [i.strip() for i in token_str[token_str.index("(")+1:].split(",")]
        normed_words = token_list[0] #x|y|z
        tag_name = token_list[1] #tags
        default = ""
        assign_value = ""
        if len(token_list) > 2:
            default = token_list[2]
        if len(token_list) > 3:
            assign_value = token_list[3][:-1]
        magic_expr = Expr(token_str, key_word, TokenType.MAGIC)
        magic_expr.default = default
        magic_expr.assign_value = assign_value
        magic_expr.tag_name = tag_name
        magic_expr.list = parse_normed_words(normed_words)
        self.add_global_token(magic_expr.key_word, magic_expr)
        self.add_token([magic_expr.key_word,magic_expr])



    def print(self):
        print ('key is', self.key_word)
        for token in self.list:
            print(token[0], end = "\t")
            if token != None:
                token[1].print()
            else:
                print ('%s is none', token)

    def eval_terminal(self, input_str):
        if input_str.startswith(self.key_word):
            match_word = self.key_word
            """
            返回匹配单词， 剩下单词， 匹配类型
            """
            state = True if match_word == input_str else False
            return [match_word, input_str[input_str.find(match_word) + len(match_word):],state, EvalType.TERMINAL]
        return None

    def eval_option_list(self, input_str):
        """
        [a b c]
        :param input_str: 
        :return: 
        """
        return self.eval_non_terminal(input_str)


    def eval_or_list(self, input_str):
        """
        a | b | c
        :param input_str: 
        :return: 
        """
        ans = []
        for token, expr in self.list:
            eval_ret = expr.eval(input_str)
            if eval_ret:
                if expr.token_type == TokenType.TERMINAL:
                    ans.append([eval_ret])
                else:
                    for ret in eval_ret:
                        ans.append(ret)

        return ans
    def get_default_match(self, input_str):
        return ['', input_str, False, TokenType.DEFAULT]
    def eval_non_terminal(self, input_str):
        """
        <a>
        :param input_str: 
        :return: 
        """
        ans = []
        for i, token_expr in enumerate(self.list):
            tmp = []
            token, expr = token_expr
            if i == 0:
                eval_ret = expr.eval(input_str)
                if not eval_ret:
                    continue
                if expr.token_type == TokenType.TERMINAL:
                    tmp.append([eval_ret])
                elif expr.token_type == TokenType.OPTION_LIST:
                    for ret in eval_ret:
                        tmp.append(ret)
                        if [self.get_default_match(input_str)] not in tmp:
                            tmp.append([self.get_default_match(input_str)])
                else:
                    for ret in eval_ret:
                        tmp.append(ret)

            else:
                for a in ans:
                    eval_ret = expr.eval(a[-1][1])
                    if not eval_ret:
                        continue
                    if expr.token_type == TokenType.TERMINAL:
                        t = deepcopy(a)
                        t.append(eval_ret)
                        tmp.append(t)
                    elif expr.token_type == TokenType.NON_TERMINAL:
                        for ret in eval_ret:
                            t = deepcopy(a)
                            t.extend(ret)
                            tmp.append(t)
                    elif expr.token_type == TokenType.OPTION_LIST:
                        for ret in eval_ret:
                            tmp.append(a)
                            t = deepcopy(a)
                            t.extend(ret)
                            tmp.append(t)
                    else:
                        for ret in eval_ret:
                            t = deepcopy(a)
                            t.append(ret)
                            tmp.append(t)
            ans = tmp

        return ans

    def eval_magic(self, input_str):
        ans = []
        for token, expr in self.list:
            eval_ret = expr.eval(input_str)
            if not eval_ret:
                continue
            if expr.token_type == TokenType.TERMINAL:
                tag_value = eval_ret[0]
                if self.assign_value:
                    assign_value = self.assign_value
                elif self.default:
                    assign_value = self.default
                else:
                    assign_value = tag_value
                eval_ret.append(tag_value)
                eval_ret.append(assign_value)
                ans.append([eval_ret])
            else:
                for ret in eval_ret:
                    tag_value = ret[0]
                    assign_value = tag_value if self.assign_value  else self.default if self.default else tag_value
                    ret.append(tag_value)
                    ret.append(assign_value)
                    ans.append(ret)
        return ans

    def eval(self, input_str):
        """
        eval expr
        表达式求值
        """
        input_str = input_str.strip()
        if self.token_type == TokenType.TERMINAL:
            return self.eval_terminal(input_str)
        elif self.token_type == TokenType.OPTION_LIST:
            return self.eval_option_list(input_str)
        elif self.token_type == TokenType.OR_LIST:
            return self.eval_or_list(input_str)
        elif self.token_type == TokenType.NON_TERMINAL:
            return self.eval_non_terminal(input_str)
        elif self.token_type == TokenType.MAGIC:
            return self.eval_magic(input_str)
        else:
            return None














