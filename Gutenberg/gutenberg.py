'''
Created on Nov 14, 2018

@author: joerg
'''
from gutenbergdammit.ziputils import loadmetadata
from gutenbergdammit.ziputils import searchandretrieve
from gutenbergdammit.ziputils import retrieve_one
import ndjson
from collections import Counter
from collections import Set
import operator

def load_ndjson(file):
    with open(file) as f:
        data = ndjson.load(f)
    return data



# for info, text in searchandretrieve("gutenberg-dammit-files-v002.zip", {'Num': '615'}):
#     print(info['Title'][0], info['Num'], len(text))


def retrieve_from_gutenberg(filename):
    book = retrieve_one("gutenberg-dammit-files-v002.zip", filename+'.txt')
    
    book =  book.splitlines()
    return book

def gutenberg_to_singlefiles():
    ndjson_gutenberg = load_ndjson('/home/joerg/workspace/thesis/gute_Daten/english_untagged/gutenberg-poetry-v001.ndjson')
    
    unique_gid = []
    for line in ndjson_gutenberg:
        if line['gid'] not in unique_gid:
            unique_gid.append(line['gid'])
    print(unique_gid)
    gutenberg_path =[]
    for gid in unique_gid:
        offset = 5 - len(str(gid))
        gid = '0'*offset + str(gid)
        gid = str(gid[:3])+'/'+str(gid)
        gutenberg_path.append(gid)
        
    for gid_path in gutenberg_path:
        book = retrieve_from_gutenberg(gid_path)
        with open(gid_path[-5:]+'.txt', 'w') as file:
            for line in book:
                file.write("%s\n" %  line)


def number_of_token2():
    ndjson_gutenberg = load_ndjson('/home/joerg/workspace/thesis/gute_Daten/english_untagged/gutenberg-poetry-v001.ndjson')
    
    keys={}
    for line in ndjson_gutenberg:
        for word in line['s'].split():
            if word in keys:
                keys[word] +=1
            else:
                keys[word] = 1
    
    sorted_by_value = sorted(keys.items(), key=lambda kv: kv[1])
    print(sorted_by_value[::-1][:100])
    
if __name__ == '__main__':
#     gutenberg_to_singlefiles()
    number_of_token2()
                
    
    