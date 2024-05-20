from datasets import load_dataset
import json
import os

n_sample = 4

def collect_corpus(dataset, modelsize):
    if dataset == 'conala':
        ori_code = load_dataset("neulab/conala")['test']
        ori_code = [item['snippet'] for item in ori_code]

        with open('../data/conala/aug-code-{}.json'.format(modelsize), 'r') as f:
            aug_code = json.load(f)['test']

    elif dataset == 'mbpp':
        dataset = []
        with open('../data/mbpp/mbpp.jsonl', 'r') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                dataset.append(js)

        ori_code = []
        for i in range(11, 511):
            ori_code.append(dataset[i]['code'])

        with open('../data/mbpp/aug-code-{}.json'.format(modelsize), 'r') as f:
            aug_code = json.load(f)['test']

    elif dataset == 'apps':
        with open('../data/apps/ori-code.json', 'r') as f:
            ori_code = json.load(f)['test']

        with open('../data/apps/aug-code-{}.json'.format(modelsize), 'r') as f:
            aug_code = json.load(f)['test']

    elif dataset == 'mbjp':
        with open('../data/mbjp/mbjp.json', 'r') as f:
            dataset = json.load(f)
        ori_code = []
        for i in range(374, len(dataset)):
            ori_code.append(dataset[i]['canonical_solution'])

        with open('../data/mbjp/aug-code-{}.json'.format(modelsize), 'r') as f:
            aug_code = json.load(f)['test']

    return ori_code, aug_code


def main():
    for dataset in ['mbpp', 'apps', 'conala', 'mbjp']:
        for modelsize in ['7b', '13b', '34b', 'gpt35']:
            ori_code, aug_code = collect_corpus(dataset, modelsize)

            # for original code
            file = []
            for i in range(len(ori_code)):
                file.append({'id': i, 'contents': ori_code[i]})
            output_dir = './index/{}/ori-code'.format(dataset)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, 'file.json'), 'w') as f:
                json.dump(file, f)

            file = []
            for i in range(len(ori_code)):
                code = ori_code[i]
                for j in range(n_sample - 1):
                    code += '\n' + ori_code[i]
                for j in range(n_sample):
                    code += '\n' + aug_code[4 * i + j]
                file.append({'id': i, 'contents': code})
            output_dir = './index/{}/aug-code-{}'.format(dataset, modelsize)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                with open(os.path.join(output_dir, 'file.json'), 'w') as f:
                    json.dump(file, f)


if __name__ == '__main__':
    main()