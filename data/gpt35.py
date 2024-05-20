import os
import openai
import argparse
import json
from tqdm import tqdm
import backoff

pl = {'conala': 'python',
      'mbpp': 'python',
      'apps': 'python',
      'mbjp': 'java'}

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--apikey", type=str, required=True)
    parser.add_argument("--save_interval", type=int, required=True)
    parser.add_argument("--file_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sum_first", action='store_true')
    parser.add_argument("--n_gen", type=int, required=True)

    # print arguments
    args = parser.parse_args()

    openai.api_key = args.apikey

    example_q, example_c = collect_icl_examples(args)
    queries = collect_queries(args)

    if args.sum_first:
        codes = collect_codes(args)
        summarize_code(args, example_q, example_c, codes)
        with open('./openai-gen/{}/sum_{}.json'.format(args.dataset, args.file_name)) as f:
            queries = json.load(f)
        args.file_name = 'sumtocode_' + args.file_name

    generate_code(args, example_q, example_c, queries)



@backoff.on_exception(backoff.constant, openai.error.RateLimitError, interval=35)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate_prompt_generation(args, example_q, example_c, input, index):
    import random
    prompt = 'please generate a {} code snippet according to the last given description. ' \
             'Only output the code snippets. Do not explain the code. I will show you four examples first.\n'.format(pl[args.dataset])
    for i in range(4):
        id = random.randint(0, len(example_q) - 1)
        prompt += 'Description:\n' + example_q[id] + '\nCode:\n' + example_c[id] + '\n\n'
    prompt += 'Description:\n' + input[index] + '\nCode:\n'

    return prompt

def generate_prompt_summarization(args, example_q, example_c, input, index):
    import random
    prompt = 'What is the main purpose of the fifth {} code snippet? ' \
             'Summarize in one sentence and be concise. I will show you four examples first.\n'.format(pl[args.dataset])
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


def summarize_code(args, example_q, example_c, codes):
    if os.path.exists('./openai-gen/{}/sum_{}.json'.format(args.dataset, args.file_name)):
        with open('./openai-gen/{}/sum_{}.json'.format(args.dataset, args.file_name), 'r') as f:
            start_index = len(json.load(f))
    else:
        start_index = 0
        with open('./openai-gen/{}/sum_{}.json'.format(args.dataset, args.file_name), 'w') as f:
            json.dump([], f)

    gen_content = []
    for i in tqdm(range(start_index, len(codes))):

        prompt = generate_prompt_summarization(args=args, example_q=example_q, example_c=example_c, input=codes, index=i)
        response = completions_with_backoff(model='gpt-3.5-turbo',
                                            messages=[{'role': 'user',
                                                       'content': prompt}],
                                            temperature=1.0,
                                            max_tokens=128,
                                            top_p=0.99,
                                            frequency_penalty=0.0,
                                            presence_penalty=0.0)
        gen_content.append(response['choices'][0]['message']['content'])

        if (i + 1) % args.save_interval == 0:
            with open('./openai-gen/{}/sum_{}.json'.format(args.dataset, args.file_name), 'r') as f:
                ori_gen_content = json.load(f)
            ori_gen_content.extend(gen_content)
            with open('./openai-gen/{}/sum_{}.json'.format(args.dataset, args.file_name), 'w') as f:
                json.dump(ori_gen_content, f)
            gen_content = []

    with open('./openai-gen/{}/sum_{}.json'.format(args.dataset, args.file_name), 'r') as f:
        ori_gen_content = json.load(f)
    ori_gen_content.extend(gen_content)
    with open('./openai-gen/{}/sum_{}.json'.format(args.dataset, args.file_name), 'w') as f:
        json.dump(ori_gen_content, f)


def generate_code(args, example_q, example_c, queries):
    if os.path.exists('./openai-gen/{}/{}.json'.format(args.dataset, args.file_name)):
        with open('./openai-gen/{}/{}.json'.format(args.dataset, args.file_name), 'r') as f:
            start_index = len(json.load(f)) // args.n_gen
    else:
        start_index = 0
        with open('./openai-gen/{}/{}.json'.format(args.dataset, args.file_name), 'w') as f:
            json.dump([], f)

    gen_content = []
    for i in tqdm(range(start_index, len(queries))):
        prompt = generate_prompt_generation(args=args, example_q=example_q, example_c=example_c, input=queries, index=i)
        response = completions_with_backoff(model='gpt-3.5-turbo',
                                            messages=[{'role': 'user',
                                                       'content': prompt}],
                                            temperature=1.0,
                                            max_tokens=256,
                                            top_p=0.99,
                                            frequency_penalty=0.0,
                                            presence_penalty=0.0,
                                            n=args.n_gen)
        for j in range(args.n_gen):
            gen_content.append(response['choices'][j]['message']['content'])

        if (i + 1) % args.save_interval == 0:
            with open('./openai-gen/{}/{}.json'.format(args.dataset, args.file_name), 'r') as f:
                ori_gen_content = json.load(f)
            ori_gen_content.extend(gen_content)
            with open('./openai-gen/{}/{}.json'.format(args.dataset, args.file_name), 'w') as f:
                json.dump(ori_gen_content, f)
            gen_content = []

    with open('./openai-gen/{}/{}.json'.format(args.dataset, args.file_name), 'r') as f:
        ori_gen_content = json.load(f)
    ori_gen_content.extend(gen_content)
    with open('./openai-gen/{}/{}.json'.format(args.dataset, args.file_name), 'w') as f:
        json.dump(ori_gen_content, f)



if __name__ == '__main__':
    main()