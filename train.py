from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gcn.utils import load_data, accuracy, load_data_original, load_data_tu, load_dgl_fraud_data, load_nifty
from gcn.models import GCN, MLPNet, GAT

from util import inform

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="cora", help='Graph Dataset.')
parser.add_argument('--model', default="mlp", help='Applied Model.')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# args.dataset = 'pubmed'
# load data
if args.dataset in ["citeseer", "cora", "pubmed"]:
    adj, features, labels, idx_train, idx_val, idx_test = load_data_original('./data/dataset/original/', args.dataset)
elif args.dataset in ['yelp', 'amazon']:
    adj, features, labels, idx_train, idx_val, idx_test = load_dgl_fraud_data(args.dataset)
elif args.dataset in ['german', 'credit']:
    adj, features, labels, idx_train, idx_val, idx_test = load_nifty(args.dataset)
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data_tu(args.dataset, args.dataset)

# args.model = "gat"
# alpha = 0
# # sim_metric = 'jaccard'
# sim_metric = 'cosine'
# if sim_metric=='jaccard':
#     # alpha = 2e-4
#     lap = inform.get_similarity_lap_matrix(adj, metric='jaccard')
# elif sim_metric == 'cosine':
#     # alpha = 1e-6
#     lap = inform.get_similarity_lap_matrix(features, metric='cosine')
# else:
#     raise ValueError('Please specify the type of similarity metric.')
# # print(lap)
# # assert False

# Model and optimizer
if args.model == "gcn":
    model = GCN(feat=features.shape[1],
                hidden_dim=args.hidden,
                n_classes=labels.max().item() + 1,
                dropout=args.dropout)
elif args.model == "gat":
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
else:
    model = MLPNet(feat=features.shape[1],
                   hidden_dim=args.hidden,
                   n_classes=labels.max().item() + 1,
                   dropout=args.dropout)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_fn = nn.CrossEntropyLoss()
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # adding the fairness enhancing item
    # loss_fair = inform.fairness_loss(lap, output)
    # print("loss_fair : ", loss_fair.detach().numpy())
    # loss_train += alpha * loss_fair
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = loss_fn(output[idx_val], labels[idx_val])

    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    output = torch.exp(output)
    loss_fn = nn.CrossEntropyLoss()
    loss_test = loss_fn(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    # loss_fair = inform.fairness_loss(lap, output)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    # breakpoint()
    return loss_test.item(), acc_test.item(), output


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test_loss, test_acc, output = test()
timestamps = str(round(time.time()))
# with open("./output/pred/%s_%s_pred_acc_%s_%s.pkl" % (args.dataset, args.model, test_acc, timestamps),
#           "wb") as wf:
with open("./output/pred/%s_%s_pred.pkl" % (args.dataset, args.model), "wb") as wf:
    wf.write(pkl.dumps(output))
