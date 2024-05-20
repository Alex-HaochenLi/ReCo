from tqdm import trange
import json
import numpy as np
from datasets import load_dataset

def collect_query(dataset, component=None, modelsize=None):
    if dataset == 'conala':
        dataset = load_dataset("neulab/conala")['test']
        ori_query = []
        for item in dataset:
            if item['rewritten_intent'] is not None:
                ori_query.append(item['rewritten_intent'])
            else:
                ori_query.append(item['intent'])

        with open('../data/conala/aug-query-{}.json'.format(modelsize), 'r') as f:
            aug_query = json.load(f)['test']

    elif dataset == 'mbpp':
        dataset = []
        with open('../data/mbpp/mbpp.jsonl', 'r') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                dataset.append(js)

        ori_query = []
        for i in range(11, 511):
            ori_query.append(dataset[i]['text'])

        with open('../data/mbpp/aug-query-{}.json'.format(modelsize), 'r') as f:
            aug_query = json.load(f)['test']

    elif dataset == 'apps':
        with open('../data/apps/ori-query.json', 'r') as f:
            ori_query = json.load(f)['test']

        with open('../data/apps/aug-query-{}.json'.format(modelsize), 'r') as f:
            aug_query = json.load(f)['test']

    elif dataset == 'mbjp':
        with open('../data/mbjp/mbjp.json', 'r') as f:
            dataset = json.load(f)
        ori_query = []
        for i in range(374, len(dataset)):
            ori_query.append(dataset[i]['description'])

        with open('../data/mbjp/aug-query-{}.json'.format(modelsize), 'r') as f:
            aug_query = json.load(f)['test']

    return ori_query, aug_query


def bm25_mrr(dataset_type, n_sample=None):
    for modelsize in ['7b', '13b', '34b', 'gpt35']:
        ori_query, aug_query = collect_query(dataset_type, component, modelsize)
        from pyserini.search.lucene import LuceneSearcher
        if component is not None:
            aug_searcher = LuceneSearcher('./index/{}/aug-{}-index'.format(dataset_type, component))
            ori_searcher = LuceneSearcher('./index/{}/ori-index'.format(dataset_type))
        else:
            ori_searcher = LuceneSearcher('./index/{}/ori-index'.format(dataset_type))
            aug_searcher = LuceneSearcher('./index/{}/aug-index-{}'.format(dataset_type, modelsize))
        ranks_ori, ranks_q, ranks_c, ranks_qc = [], [], [], []
        for i in range(len(ori_query)):
            if type(aug_query[i]) == list:
                aug_query[i] = ' '.join(aug_query[i])
            hits_ori = ori_searcher.search(ori_query[i], k=1000)

            query = ori_query[i]
            for j in range(n_sample - 1):
                query += '\n' + ori_query[i]
            for j in range(n_sample):
                query += '\n' + aug_query[4 * i + j]

            hits_q = ori_searcher.search(query, k=1000)
            hits_qc = aug_searcher.search(query, k=1000)

            ranks_ori.append(0)
            ranks_q.append(0)
            ranks_qc.append(0)
            for idx in range(len(hits_ori)):
                if hits_ori[idx].docid == str(i):
                    ranks_ori[-1] = (1 / (idx + 1))

            for idx in range(len(hits_q)):
                if hits_q[idx].docid == str(i):
                    ranks_q[-1] = (1 / (idx + 1))

            for idx in range(len(hits_qc)):
                if hits_qc[idx].docid == str(i):
                    ranks_qc[-1] = (1 / (idx + 1))

        print('Dataset:', dataset_type + ' ' + modelsize)
        print('Default:', float(np.mean(ranks_ori)))
        print('GAR:', float(np.mean(ranks_q)))
        print('ReCo:', float(np.mean(ranks_qc)))


if __name__ == '__main__':
    n_sample = 4
    for dataset_type in ['conala', 'mbpp', 'apps', 'mbjp']:
        bm25_mrr(dataset_type, n_sample)
