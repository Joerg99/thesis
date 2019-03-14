'''
Created on Nov 24, 2018

@author: joerg
'''
from interface import Corpus
import itertools


rhymes = []
# temp = '1231'
# with open('all_rhyme.pgold', 'r') as file:
#     for line in file:
#         if line == temp:
#             continue
#         else:
#             rhymes.append(line)
#             temp = line
count = 0
overall = 0
with open('all_rhyme.pgold', 'r') as file:
    for line in file:
        if not line:
            continue
        count+= 1
        overall += 1
        if not count == 4:
            if overall < 19000:
                rhymes.append(line.strip())
        else:
            count = 0

rhymes  = [line for line in rhymes if not line == '']

rhymes = [l.split() for l in rhymes]
alphas = []
digits = []

for i in range(len(rhymes)):
    if i %2 == 0:
        rhymes[i] = rhymes[i][1:]
        alphas.append(rhymes[i])
    else:
        digits.append(rhymes[i])

zips = []
for i in range(len(digits)):
    zips.append(list(zip(alphas[i], digits[i])))
train_data = []
for l in zips:
    for a,b in itertools.combinations(l,2):
        if a[1] == b[1]:
            train_data.append((a[0]+'\t'+b[0]+'\t'+'y'))
#         else:
#             train_data.append((a[0]+'\t'+b[0]+'\t'+'n'))

with open('/home/joerg/workspace/deep-siamese-text-similarity/train_data.tsv', 'w') as file:
    for line in train_data:
        file.write('%s\n' % line)

# print(alphas[:10])
# print(digits[:10])