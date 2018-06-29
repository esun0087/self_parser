import re
class Type(object):
    def __init__(self, commit):
        self.commit = commit


class WordType:
    WORD = Type("word")
    LIST = Type("list")
    OR_LIST = Type("or_list")
    OPT_LIST = Type("opt_list")
    REF = Type("ref")
    MAGIC = Type("magic")


class WorldToken:
    def get_word_extend(self):
        m_n_re  = re.compile(r"{(?P<m>\d*),(?P<n>\d*)}")
        m_n_group = m_n_re.search(self.value)
        if m_n_group:
            m = int(m_n_group.group("m")) if m_n_group.group("m") != "" else 0
            n = int(m_n_group.group("n")) if m_n_group.group("n") != "" else 0

            re_part = self.value[:m_n_group.start()]
            if m > 0 and n > 0:
                self.value_extend = [re.compile("%s{%s,%s}" % (re_part, i, i)) for i in range(m ,n+1)]
            elif m > 0  and  n == 0:
                n  = 20
                self.value_extend = [re.compile("%s{%s,%s}" % (re_part, i, i)) for i in range(m, n + 1)]
            elif m == 0 and n > 0:
                self.value_extend = [re.compile("%s{%s,%s}" % (re_part, i, i)) for i in range(m, n + 1)]
        pass

    def __init__(self, v):
        self.value = v
        self.value_extend = None
        self.re = re.compile(v, re.I)
        self.type = WordType.WORD
        self.get_word_extend()
    def parse_extend(self,sent):
        ans = []
        for v in self.value_extend:
            re_ret = v.match(sent)
            if re_ret:
                ans.append(((re_ret.group(),), sent[len(re_ret.group()):]))
            else:
                ans.append((None, sent))
        return ans

    def parse(self, sent):
        sent = sent.strip()
        if self.value_extend:
            return self.parse_extend( sent)
        re_ret = self.re.match(sent)
        if re_ret:
            return [((re_ret.group(),), sent[len(re_ret.group()):])]
        else :
            return [(None, sent)]
    def __str__(self):
        return self.value

class MagicMatchResult:
    def __init__(self, magic_field, magic_value, magic_norm_value):
        self.magic_field = magic_field
        self.magic_value = magic_value
        self.magic_norm_value = magic_norm_value
    def __str__(self):
        return "%s_%s_%s" % (self.magic_field, self.magic_value, self.magic_norm_value)
    def __repr__(self):
        return "%s_%s_%s" % (self.magic_field, self.magic_value, self.magic_norm_value)

class MagicToken:
    def __init__(self, v, replace_token=None, default=''):
        self.v = v
        self.type = WordType.MAGIC
        self.default = default
        self.replace_token = replace_token
        self.match_value = None

    def parse(self, sent):
        match_result = self.replace_token.parse(sent)
        self.match_value = [((MagicMatchResult(self.v, i[0], i[0] if self.default.value == 'default' else self.default),),j) for i, j in match_result if i and i[0]]
        return self.match_value





class List_Token:
    def __init__(self, v):
        self.value = v
        self.type = WordType.LIST

    def parse(self, sent):
        ans = None

        for i, t in enumerate(self.value):
            if i == 0:
                match_list = t.parse(sent)
                ans = match_list
            else:
                ans1 = []
                for a in ans:
                    left, right = a[0], a[1]
                    match_list = t.parse(right)
                    for ll, rr in match_list:
                        if ll:
                            ans1.append((list(left) + list(ll) if ll[0] else list(left), rr))
                ans = ans1
        return ans


class Or_List_Token:
    def __init__(self, v):
        self.value = v
        self.type = WordType.OR_LIST
    def parse(self, sent):
        ans = []
        for v in self.value:
            match_list = v.parse(sent)
            for ll, rr in match_list:
                if ll:
                    ans.append((ll if ll[0] else [], rr))
        return ans

class Opt_List_Token:
    def __init__(self, v):
        self.value = v
        self.type = WordType.OPT_LIST
    def parse(self, sent):
        ans = None

        for i, t in enumerate(self.value):
            if i == 0:
                match_list = t.parse(sent)
                ans = match_list
            else:
                ans1 = []
                for a in ans:
                    left, right = a[0], a[1]
                    match_list = t.parse(right)
                    for ll, rr in match_list:
                        if ll:
                            ans1.append((list(left) + list(ll) if ll[0] else list(left), rr))
                ans = ans1
        ans.append((("",), sent))
        return ans



class Ref_Token:
    def __init__(self, v, def_dic):
        self.value = v
        self.type = WordType.REF
        self.match_value = None
        self.def_dic = def_dic
    def parse(self, sent):
        ref_def = self.def_dic[self.value.value] if self.value.value in self.def_dic else None
        match_result = ref_def.parse(sent)
        self.match_value = match_result
        return match_result




