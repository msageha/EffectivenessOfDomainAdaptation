import re

def is_num(text):
    m = re.match('\A[0-9]+\Z', text)
    if m:
        return True
    else:
        return False
