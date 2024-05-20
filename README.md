

# Rewriting the Code: A Simple Framework for Large Language Model Augmented Semantic Code Search

This repo contains code for 
[Rewriting the Code: A Simple Framework for Large Language Model Augmented Semantic Code Search](https://arxiv.org/abs/2401.04514), accepted to ACL 2024.
In this codebase we provide instructions for reproducing our results from the paper.
We hope that this work can be useful for future research on 
Generation-Augmented Retrieval framework for code search.


## Environment
```angular2html
conda create -n ReCo python=3.8 -y
conda activate ReCo
conda install pytorch-gpu=1.7.1 -y
pip install transformers datasets tqdm tree-sitter openai fairscale
fire sentencepiece backoff edit_distance pyserini
```

## Data
For the detailed information of data we used in our experiments,
please refer to [README.md](data/README.md) in `./data`.

## ReCo
For the detailed information of ReCo and GAR in our paper, please refer to 
[README.md](ReCo/README.md) in `./ReCo`.

## Metrics
For the detailed information of Code Style Distance in our paper, please refer to 
[README.md](metrics/README.md) in `./metrics`.

## Citation
If you found this repository useful, please consider citing:
```bibtex
@article{li2024rewriting,
  title={Rewriting the Code: A Simple Method for Large Language Model Augmented Code Search},
  author={Li, Haochen and Zhou, Xin and Shen, Zhiqi},
  journal={arXiv preprint arXiv:2401.04514},
  year={2024}
}
```