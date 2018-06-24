#coding:utf-8
"""
设置基本的类型
"""

class BaseType(object):
    def __init__(self, value, commit):
        self.value = value
        self.commit = commit

class TokenType(object):
    NON_TERMINAL = BaseType(1, "token_non_terminal")
    TERMINAL = BaseType(2, "token_terminal")
    OR_LIST = BaseType(3, "token_or_list")
    OPTION_LIST = BaseType(4, "token_option_list")
    MAGIC = BaseType(5, "token_magic")
    DEFAULT = BaseType(6, "token_default")
    INVALID = BaseType(-1, "token_invalid")

class EvalType(object):
    NON_TERMINAL = BaseType(1, "eval_non_terminal")
    TERMINAL = BaseType(2, "eval_terminal")
    OR_LIST = BaseType(3, "eval_or_list")
    OPTION_LIST = BaseType(4, "eval_option_list")
    MAGIC = BaseType(5, "eval_magic")
    INVALID = BaseType(-1, "eval_invalid")


