

# ReCo
In this folder, we provide Python files that could reproduce the main experiments
in our paper.

## BM25
`bm25_pipline.sh` contains the steps used for BM25 evaluation.

It firstly change the codebase into the Pyserini format, then index the codebase,
and finally run `bm25.py` for evaluation. This script will automatically
evaluate BM25 with four LLMs on four datasets using four LLM-generated codes.

## Zero-shot
### UniXcoder
To evaluate UniXcoder under the zero-shot setting, run:
```angular2html
dataset='mbpp'
model='7b'
python run.py --model_name_or_path microsoft/unixcoder-base \
--do_eval --code_length 512 --nl_length 512 \ 
--eval_batch_size 64 --seed 123456 \
--query_gen_path ./data/${dataset}/aug-query-${model}.json \
--code_gen_path ./data/${dataset}/aug-code-${model}.json \
--n_gen 4 --max_n_gen 4 --augquery --augcode --dataset_type ${dataset}
```

This script evaluates UniXcoder with ReCo on MBPP dataset using 4 Code Llama-7b
generated and rewritten codes. To evaluate UniXcoder with GAR, simply remove `--augcode` in the argument.
To change the number of exemplar and rewritten codes used in ReCo or GAR, revise the number in `--n_gen`.
To change LLMs, choose `model` from `[7b, 13b, 34b, gpt35]`.
To change datasets, choose `dataset` from `[conala, mbpp, apps, mbjp]`.

### Contriever

To evaluate using Contriever, simply replace `microsoft/unixcoder-base` in `--model_name_or_path`
 to `facebook/contriever-msmarco`.

## Fine-tune
### UniXcoder
To fine-tune and evaluate UniXcoder, run:
```angular2html
dataset='mbpp'
model='7b'
python run.py --output_dir ./saved_models/${save_dir} \
--model_name_or_path microsoft/unixcoder-base \
--do_train --num_train_epochs 10 --code_length 256 --nl_length 256 \
--train_batch_size 32 --eval_batch_size 64 \
--learning_rate 5e-6 --seed 123456 \
--query_gen_path ./data/${dataset}/aug-query-${modelsize}.json \
--code_gen_path ./data/${dataset}/aug-code-${modelsize}.json \
--n_gen 4 --max_n_gen 4 --augcode --augquery --dataset_type ${dataset}
```

### CodeBERT
To fine-tune and evaluate UniXcoder, simply replace `microsoft/unixcoder-base` in `--model_name_or_path`
 to `microsoft/codebert-base`. 

As mentioned in our paper, CodeBERT is first pre-trained on CodeSearchNet.
We use `--ckpt` to add the path of pre-trained checkpoint.

