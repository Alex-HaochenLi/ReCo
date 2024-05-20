# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import UniXcoderModel, CodeBERTModel, CodeT5pModel
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BertConfig, BertModel, BertTokenizer,
                          AutoConfig, AutoModel, AutoTokenizer)
from datasets import load_dataset

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids

        
def convert_examples_to_features(js,tokenizer,args):
    """convert examples to token ids"""
    code = js['code']
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = js['query']
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, phase=None):
        self.examples = []
        self.data = []

        if args.dataset_type == 'mbpp':
            dataset = []
            with open('../data/mbpp/mbpp.jsonl', 'r') as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    dataset.append(js)
            if phase == 'train':
                for i in range(601, len(dataset)):
                    js = {'query': dataset[i]['text'], 'code': dataset[i]['code']}
                    self.data.append(js)
            elif phase == 'test':
                for i in range(11, 511):
                    js = {'query': dataset[i]['text'], 'code': dataset[i]['code']}
                    self.data.append(js)
        elif args.dataset_type == 'mbjp':
            with open('../data/mbjp/mbjp-filtered.json', 'r') as f:
                dataset = json.load(f)
            if phase == 'train':
                for i in range(374):
                    js = {'query': dataset[i]['description'], 'code': dataset[i]['canonical_solution']}
                    self.data.append(js)
            elif phase == 'test':
                for i in range(374, len(dataset)):
                    js = {'query': dataset[i]['description'], 'code': dataset[i]['canonical_solution']}
                    self.data.append(js)
        elif args.dataset_type == 'apps':
            with open('../data/apps/ori-query.json', 'r') as f:
                query = json.load(f)[phase]
            with open('../data/apps/ori-code.json', 'r') as f:
                codes = json.load(f)[phase]
            for i in range(len(query)):
                js = {'query': query[i], 'code': codes[i]}
                self.data.append(js)
        elif args.dataset_type == 'conala':
            dataset = load_dataset("neulab/conala")[phase]
            for i in range(len(dataset)):
                if dataset[i]['rewritten_intent'] is not None:
                    js = {'query': dataset[i]['rewritten_intent'], 'code': dataset[i]['snippet']}
                else:
                    js = {'query': dataset[i]['intent'], 'code': dataset[i]['snippet']}
                self.data.append(js)

        for js in self.data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),
                torch.tensor(self.examples[i].nl_ids),
                i)


class augTextDataset(Dataset):
    def __init__(self, tokenizer, args, phase=None, query_gen_path=None, code_gen_path=None):
        self.examples = []
        self.args = args
        self.data = []

        with open(query_gen_path, 'r') as f:
            query_gen_aug = json.load(f)
            query_gen_aug = query_gen_aug[phase]
        with open(code_gen_path, 'r') as f:
            code_gen_aug = json.load(f)
            code_gen_aug = code_gen_aug[phase]
        assert len(query_gen_aug) == len(code_gen_aug)
        for i in range(len(query_gen_aug)):
            js = {'query': query_gen_aug[i], 'code': code_gen_aug[i]}
            self.data.append(js)

        for js in self.data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))

    def __len__(self):
        return len(self.examples) // self.args.n_gen

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))

            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer, config):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, phase='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    aug_dataset = augTextDataset(tokenizer, args, phase='train', query_gen_path=args.query_gen_path,
                                 code_gen_path=args.code_gen_path)

    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0
    for idx in range(args.num_train_epochs): 
        for step, batch in enumerate(train_dataloader):

            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)

            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)

            aug_index = batch[2]
            if args.augcode:
                code_inputs = torch.cat(
                    [torch.tensor(aug_dataset.examples[i * args.max_n_gen + j].code_ids).unsqueeze(0) for i in aug_index
                     for j in range(args.n_gen)], dim=0).to(args.device)
                code_vec2 = model(code_inputs=code_inputs)
                code_vec2 = code_vec2.view(batch[1].size(0), args.n_gen, config.hidden_size)
                code_vec = torch.mean(torch.cat([code_vec.unsqueeze(1).repeat(1, args.n_gen, 1), code_vec2], dim=1),
                                      dim=1)

            if args.augquery:
                nl_inputs = torch.cat(
                    [torch.tensor(aug_dataset.examples[i * args.max_n_gen + j].nl_ids).unsqueeze(0) for i in aug_index for
                     j in range(args.n_gen)], dim=0).to(args.device)
                nl_vec2 = model(nl_inputs=nl_inputs)
                nl_vec2 = nl_vec2.view(batch[1].size(0), args.n_gen, config.hidden_size)
                nl_vec = torch.mean(torch.cat([nl_vec.unsqueeze(1).repeat(1, args.n_gen, 1), nl_vec2], dim=1), dim=1)

            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores*20, torch.arange(batch[0].size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%(len(train_dataloader) // 3) == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer, config)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, config):
    test_dataset = TextDataset(tokenizer, args, phase='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4)

    aug_dataset = augTextDataset(tokenizer, args, phase='test', query_gen_path=args.query_gen_path,
                               code_gen_path=args.code_gen_path)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num data = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = []
    nl_vecs = []
    for batch in test_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            code_vec = model(code_inputs=code_inputs)

        aug_index = batch[2]
        if args.augcode:
            code_inputs = torch.cat(
                [torch.tensor(aug_dataset.examples[i * args.max_n_gen + j].code_ids).unsqueeze(0) for i in aug_index for j in
                 range(args.n_gen)], dim=0).to(args.device)
            with torch.no_grad():
                code_vec2 = model(code_inputs=code_inputs)
                code_vec2 = code_vec2.view(batch[1].size(0), args.n_gen, config.hidden_size)
                code_vec = torch.mean(torch.cat([code_vec.unsqueeze(1).repeat(1, args.n_gen, 1), code_vec2], dim=1), dim=1)

        if args.augquery:
            nl_inputs = torch.cat(
                [torch.tensor(aug_dataset.examples[i * args.max_n_gen + j].nl_ids).unsqueeze(0) for i in aug_index for j in
                 range(args.n_gen)], dim=0).to(args.device)
            with torch.no_grad():
                nl_vec2 = model(nl_inputs=nl_inputs)
                nl_vec2 = nl_vec2.view(batch[1].size(0), args.n_gen, config.hidden_size)
                nl_vec = torch.mean(torch.cat([nl_vec.unsqueeze(1).repeat(1, args.n_gen, 1), nl_vec2], dim=1), dim=1)

        nl_vecs.append(nl_vec.cpu().numpy())
        code_vecs.append(code_vec.cpu().numpy())


    model.train()
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    scores = np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

    ranks = []
    for idx in range(len(scores)):
        rank = sort_ids[idx].tolist()
        ranks.append(1 / (rank.index(idx) + 1))

    result = {
        "eval_mrr": float(np.mean(ranks))
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--query_gen_path', type=str, default=None,
                        help="Path for loading exemplar code.")
    parser.add_argument('--code_gen_path', type=str, default=None,
                        help="Path for loading rewritten code.")
    parser.add_argument('--ckpt', type=str, default=None,
                        help="Path for loading CodeSearchNet pre-trained CodeBERT.")
    parser.add_argument('--n_gen', type=int, default=1,
                        help="The number of LLM-generated codes used for training/evaluation.")
    parser.add_argument('--max_n_gen', type=int, default=4,
                        help="The number of LLM-generated codes for each query in the data file.")
    parser.add_argument("--augcode", action='store_true',
                        help="Whether to use rewritten code.")
    parser.add_argument("--augquery", action='store_true',
                        help="Whether to use exemplar code.")
    parser.add_argument('--dataset_type', type=str, default=None,
                        help="The used dataset (conala/apps/mbpp/mbjp)")

    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    if 'codebert' in args.model_name_or_path or 'unixcoder' in args.model_name_or_path:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        model = RobertaModel.from_pretrained(args.model_name_or_path)
    elif 'contriever' in args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        config = BertConfig.from_pretrained(args.model_name_or_path)
        model = BertModel.from_pretrained(args.model_name_or_path)
    elif 'codet5p' in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        config.hidden_size = config.embed_dim

    if 'codebert' in args.model_name_or_path:
        model = CodeBERTModel(model)
    elif 'unixcoder' in args.model_name_or_path or 'contriever' in args.model_name_or_path:
        model = UniXcoderModel(model)
    elif 'codet5p' in args.model_name_or_path:
        model = CodeT5pModel(model)

    if args.ckpt is not None:
        # model_to_load = model.module if hasattr(model, 'module') else model
        model.load_state_dict(torch.load(os.path.join(args.ckpt, 'checkpoint-best-mrr/model.bin')))
    logger.info("Training/evaluation parameters %s", args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)  
            
    # Training
    if args.do_train:
        train(args, model, tokenizer, config)
      
    # Evaluation
    if args.do_eval:
        model.to(args.device)
        result = evaluate(args, model, tokenizer, config)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))


if __name__ == "__main__":
    main()
