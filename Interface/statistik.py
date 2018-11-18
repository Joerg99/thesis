'''
Created on Nov 17, 2018

@author: joerg
'''

import ndjson
from collections import Counter

def load_data(filename):
    with open(filename, 'r') as file:
        data = ndjson.load(file)
    return data


if __name__ =='__main__':
    print('ads')
    c = Counter()
#     d = load_data('/home/joerg/workspace/thesis/Gutentag/gutentag_de.ndjson')
    d = load_data('/home/joerg/workspace/thesis/Textgrid/textgrid_l_in_lg_tags_verse_types_re.ndjson')
    print(len(d))
    l = []
    for e in d:
        l.append(e['poem_no'])
    print(l[:23])
    l = [int(element) for element in l]
    s = set(l)
    l = list(s)
    l.sort()
    print('len: ', len(l))
    print(l[-10:])

