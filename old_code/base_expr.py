#coding:utf-8

"""
用来表述最基本的表达式
<a>:b c d <e>
表示上下文无关文法
"""

class Expr(object):
    def __init__(self, value):
        self.expr_list = []
        self.value = value
    def add_expr(self, expr):
        self.expr_list.append(expr)