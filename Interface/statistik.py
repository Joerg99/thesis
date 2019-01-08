'''
Created on Nov 17, 2018

@author: joerg
'''

import ndjson
from collections import Counter
import re


def load_data(filename):
    with open(filename, 'rb') as file:
        data = ndjson.load(file)
    return data

def kill_bad_sings():
    lines = []
    with open('/home/joerg/workspace/thesis/Chicago/chicago.ndjson', 'r', encoding='utf-8') as file:
        for line in file:
            if re.search(r'\\', line):
                print(line)
            
            lines.append(line)
        print(lines[9740])
    with open('/home/joerg/workspace/thesis/Chicago/chicago_clean.ndjson','w') as file:
        for line in lines:
            file.write(line)


if __name__ =='__main__':
    print('x')
#     kill_bad_sings()
#     print(data[9228:9232])
