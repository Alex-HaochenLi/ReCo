# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import os
import argparse
import json
import fire
from tqdm import tqdm
from llama import Llama
import random
import numpy as np
import torch


pl = {'conala': 'python',
      'mbpp': 'python',
      'apps': 'python',
      'mbjp': 'java'}


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sum_first", action='store_true')
    parser.add_argument("--gen_time", type=int, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--max_batch_size", type=int, required=True)

    args = parser.parse_args()
    set_seed(1)

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    example_q, example_c = collect_icl_examples(args)
    queries = collect_queries(args)
    if args.sum_first:
        codes = collect_codes(args)
        summarize_code(args, example_q, example_c, codes, generator)
        with open('./codellama-gen/{}/sum_{}.json'.format(args.dataset, args.file_name)) as f:
            queries = json.load(f)
        args.file_name = 'sumtocode_' + args.file_name

    generate_code(args, example_q, example_c, queries, generator)


def summarize_code(args, example_q, example_c, codes, generator):
    gen_content = []
    for i in tqdm(range(len(codes))):
        prompt = generate_prompt_summarization(args=args, example_q=example_q, example_c=example_c, input=codes, index=i)

        system_prompt = 'What is the main purpose of the fifth {} code snippet? ' \
         'Summarize in one sentence and be concise. I will show you four examples first.\n'.format(pl[args.dataset])
        instructions = [
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        ]

        results = generator.chat_completion(
            instructions,  # type: ignore
            max_gen_len=128,
            temperature=1.0,
            top_p=0.9,
        )

        gen_content.append(results[0]['generation']['content'])

    with open('./codellama-gen/{}/sum_{}.json'.format(args.dataset, args.file_name), 'w') as f:
        json.dump(gen_content, f)


def generate_code(args, example_q, example_c, queries, generator):
    gen_content = []
    for i in tqdm(range(len(queries))):
        for j in range(args.gen_time):
            prompt = generate_prompt_generation(args=args, example_q=example_q, example_c=example_c, input=queries, index=i)

            system_prompt = 'please generate a {} code snippet according to the fifth description. ' \
             'Only output the code snippets. Do not explain the code. I will show you four examples first.\n'.format(pl[args.dataset])
            instructions = [
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            ]

            results = generator.chat_completion(
                instructions,  # type: ignore
                max_gen_len=256,
                temperature=1.0,
                top_p=0.9,
            )

            gen_content.append(results[0]['generation']['content'])

    with open('./codellama-gen/{}/{}.json'.format(args.dataset, args.file_name), 'w') as f:
        json.dump(gen_content, f)


def generate_prompt_generation(args, example_q, example_c, input, index):
    import random
    prompt = ''
    for i in range(4):
        id = random.randint(0, len(example_q) - 1)
        prompt += 'Description:\n' + example_q[id] + '\nCode:\n' + example_c[id] + '\n\n'
    prompt += 'Description:\n' + input[index] + '\nCode:\n'

    return prompt


def generate_prompt_summarization(args, example_q, example_c, input, index):
    import random
    prompt = ''
    for i in range(4):
        id = random.randint(0, len(example_q) - 1)
        prompt += 'Code:\n' + example_c[id] + '\nPurpose:\n' + example_q[id] + '\n\n'
    prompt += 'Code:\n' + input[index] + '\nPurpose:\n'

    return prompt


def collect_icl_examples(args):
    if args.dataset == 'mbpp':
        example_q, example_c = [], []
        with open('./mbpp/mbpp.jsonl', 'r') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                example_q.append(js['text'])
                example_c.append(js['code'])
            example_c = example_c[601:]
            example_q = example_q[601:]

    elif args.dataset == 'apps':
        with open('./apps/ori-query.json', 'r') as f:
            example_q = json.load(f)
            example_q = example_q['train']
        with open('./apps/ori-code.json', 'r') as f:
            example_c = json.load(f)
            example_c = example_c['train']

    elif args.dataset == 'mbjp':
        with open('./mbjp/mbjp.json', 'r') as f:
            dataset = json.load(f)
            example_q = [item['description'] for item in dataset]
            example_c = [item['canonical_solution'] for item in dataset]
            example_c = example_c[:374]
            example_q = example_q[:374]

    elif args.dataset == 'conala':
        example_q, example_c, dataset = [], [], []
        from datasets import load_dataset
        ori_dataset = load_dataset("neulab/conala")
        for item in ori_dataset['train']:
            dataset.append(item)
        for i in range(len(dataset)):
            if dataset[i]['rewritten_intent'] is not None:
                example_q.append(dataset[i]['rewritten_intent'])
            else:
                example_q.append(dataset[i]['intent'])
            example_c.append(dataset[i]['snippet'])

    return example_q, example_c


def collect_queries(args):
    if args.dataset == 'conala':
        dataset = []
        from datasets import load_dataset
        ori_dataset = load_dataset("neulab/conala")
        for phase in ['train', 'test']:
            for item in ori_dataset[phase]:
                dataset.append(item)
        queries = []
        for i in range(len(dataset)):
            if dataset[i]['rewritten_intent'] is not None:
                queries.append(dataset[i]['rewritten_intent'])
            else:
                queries.append(dataset[i]['intent'])

    elif args.dataset == 'mbpp':
        queries = []
        with open('./mbpp/mbpp.jsonl', 'r') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                queries.append(js['text'])
            queries = queries[11: 511] + queries[601: 975]  # 11 ~ 511: test, 601 ~ 975: train

    elif args.dataset == 'apps':
        with open('./apps/ori-query.json', 'r') as f:
            queries = json.load(f)
        queries = queries['train'] + queries['test']

    elif args.dataset == 'mbjp':
        with open('./mbjp/mbjp.json', 'r') as f:
            dataset = json.load(f)
        queries = [item['description'] for item in dataset]

    return queries


def collect_codes(args):
    if args.dataset == 'conala':
        dataset = []
        from datasets import load_dataset
        ori_dataset = load_dataset("neulab/conala")
        for phase in ['train', 'test']:
            for item in ori_dataset[phase]:
                dataset.append(item)
        codes = []
        for i in range(len(dataset)):
            codes.append(dataset[i]['snippet'])

    elif args.dataset == 'mbpp':
        codes = []
        with open('./mbpp/mbpp.jsonl', 'r') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                codes.append(js['code'])
            codes = codes[11: 511] + codes[601: 975]

    elif args.dataset == 'apps':
        with open('./apps/ori-code.json', 'r') as f:
            codes = json.load(f)
        codes = codes['train'] + codes['test']

    elif args.dataset == 'mbjp':
        with open('./mbjp/mbjp.json', 'r') as f:
            dataset = json.load(f)
        codes = [item['canonical_solution'] for item in dataset]

    return codes


if __name__ == "__main__":
    fire.Fire(main)
