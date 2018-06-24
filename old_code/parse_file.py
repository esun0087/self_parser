#coding:utf-8
"""
解析配置文件
"""
from old_code.DomainExpr import  DomainExpr

if __name__ == '__main__':
    domain_expr = DomainExpr()
    domain_expr.parse_file("test_file.txt")
    # for k,v in domain_expr.all_token.items():
        # print (k, v.token_type.commit)
    ans = domain_expr.eval("abcdef")
    for each_root in ans:
        for each_res in each_root:
            print (each_res)


