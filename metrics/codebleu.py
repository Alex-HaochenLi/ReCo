# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import bleu
import weighted_ngram_match
import syntax_match
import dataflow_match


def cal_codebleu_for_one_pair(lang, hypothesis, references):
    alpha,beta,gamma,theta = 0.25,0.25,0.25,0.25

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open('keywords/'+lang+'.txt', 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token:1 if token in key_word_list else 0.2 \
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    # print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
    #                     format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score
    print('CodeBLEU score: ', code_bleu_score)

