import random
import json
from tqdm import tqdm
from datasets import load_dataset

dataset = load_dataset('codeparrot/apps')
trainset = dataset['train']
testset = dataset['test']

train_question, test_question = [], []
train_codes, test_codes = [], []

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/unixcoder-base')

for i, item in tqdm(enumerate(trainset)):
   if len(tokenizer.tokenize(item['question'].split('-----Input-----')[0])) > 512:
      continue
   count = 0
   skip = False
   while True:
      sample = random.choice(json.loads(item['solutions']))
      count += 1
      if len(tokenizer.tokenize(sample)) < 512:
         break
      if count >= 20:
         skip = True
         break
   if not skip:
      train_codes.append(sample)
      train_question.append(item['question'].split('-----Input-----')[0])

print('===================')

from json import JSONDecodeError
for i, item in tqdm(enumerate(testset)):
   try:
      if len(tokenizer.tokenize(item['question'].split('-----Input-----')[0])) > 512:
         continue
      count = 0
      skip = False
      while True:
         sample = random.choice(json.loads(item['solutions']))
         count += 1
         if len(tokenizer.tokenize(sample)) < 512:
            break
         if count >= 20:
            skip = True
            break
      if not skip:
         test_codes.append(sample)
         test_question.append(item['question'].split('-----Input-----')[0])
   except JSONDecodeError:
      continue


# len_list = []
# for question in train_question:
#    len_list.append(len(tokenizer.tokenize(question)))
# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(len_list, bins=[100,200,300,400,500,600,700,800,900,1000])
# plt.show()
# import numpy as np
# print(np.mean(len_list))

with open('./ori-query2.json', 'w') as f:
   json.dump({'train': train_question, 'test': test_question}, f)
with open('./ori-code2.json', 'w') as f:
   json.dump({'train': train_codes, 'test': test_codes}, f)




