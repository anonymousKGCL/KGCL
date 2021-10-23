## KGCL

This is the Pytorch implementation for the paper: Knowledge Graph Contrastive Learning for Recommendation, with CF learning part based on [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch)

## Enviroment Requirement

`pip install -r requirements.txt`

## Dataset

We provide three processed datasets: Yelp2018 and Amazon-book and MIND.

## An example to run KGCL

run KGCL on **Yelp2018** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command

` cd code && python main.py --dataset="yelp2018" `