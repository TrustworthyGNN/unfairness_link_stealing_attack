# unfairness in link stealing attack

## Step -1: Requirement

We follows the instruction on https://github.com/tkipf/gcn to install gcn.

## Step 0: Datasets

We provide the datasets used in our paper: 

```bash
["AIDS", "COX2", "DHFR", "ENZYMES", "PROTEINS_full", "citeseer", "cora", "pubmed"]
```

## Step 1: Train Target Model

Train GCN and MLP for all datasets.

Train GCN: 

```bash
python3 train.py --dataset cora --model gcn
```

Train MLP: 

```bash
python3 train.py --dataset cora --model mlp
```

## Step 2: Generate Partial Graph
Split train / test for all datasets.

```bash
python3 partial_graph_generation.py --dataset cora
```

## Step 3: Run Attacks
Note that all attacks rely on the partial graphs generated by Step 2.
### Attack 0
```bash
python3 attack.py --dataset cora
```

### Note
Note that in the code we use standard scaler to fit the training data and testing data separately as we assume that the adversary may acquire the whole testing data at the beginning. 

```
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
```

but please feel free to modify it into:
```
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
```

This is a Pytorch implementation of Stealing links from Graph Neural Networks, as described in our paper:

Xinlei He, Jinyuan Jia, Michael Backes, Neil Zhenqiang Gong, Yang Zhang, [Stealing Links from Graph Neural Networks](https://arxiv.org/abs/2005.02131) (Usenix Security 2021)
