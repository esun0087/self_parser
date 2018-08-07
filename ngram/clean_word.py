import codecs
import string
import  re
data = [i.strip() for i in codecs.open("song_name.txt", "r", "utf-8").readlines()]
hanzi = re.compile(r"[\u4e00-\u9fa5]+")

def filter_line(line):
    if '(' in line or ')' in line:
        return False
    for i in string.digits:
        if i in line:
            return False
    for i in string.ascii_letters:
        if i in line:
            return False
    if ' ' in line:
        return False
    if '》' in line:
        return False
    if ":" in line:
        return False
    if len(line) == 1:
        return False
    if not hanzi.match(line):
        return False
    return True


data = [i for i in data if filter_line(i)]
print (len(data))
s = set()
with codecs.open("new_song_name.txt", "w", "utf-8") as f:
    for l in data:
        if l not in s:
            f.write(l + "\n")
        s.add(l)