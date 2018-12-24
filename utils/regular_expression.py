import re


def is_num(text):
    m = re.match('\A[0-9]+\Z', text)
    if m:
        return True
    else:
        return False


def extraction_num(text):
    m = re.search(r'([-0-9]+)', text)
    return int(m.group(1))
