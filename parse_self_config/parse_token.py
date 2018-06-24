from get_token import SemanticParser

if __name__ == '__main__':
    g = SemanticParser()
    # g.parse_expr_by_token("root=magic([a | aaabb | aaabbcc | cc | <a>], a, sddd)")
    g.parse_file("arith.expr")
    ans = g.parse("1.323234加2乘3")
    for i in ans:
        print( i)
