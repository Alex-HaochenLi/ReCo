#!/bin/sh

python prepare_bm25_data.py

for dataset in 'mbpp' 'mbjp' 'apps' 'conala'
do
for modelsize in '7b' '13b' '34b' 'gpt35'
do

python -m pyserini.index.lucene --collection JsonCollection --input ./index/${dataset}/aug-code-${modelsize} --index ./index/${dataset}/aug-index-${modelsize} --generator DefaultLuceneDocumentGenerator --threads 1

done
done

python bm25.py