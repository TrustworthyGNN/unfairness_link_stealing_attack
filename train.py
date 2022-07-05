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

from gcn.utils import load_data, accuracy
from gcn.models import GCN, MLPNet

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="cora", help='Graph Dataset.')
parser.add_argument('--model', default="mlp", help='Applied Model.')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
if args.model == "gcn":
    model = GCN(feat=features.shape[1],
                hidden_dim=args.hidden,
                n_classes=labels.max().item() + 1,
                dropout=args.dropout)

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
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    # loss_val = F.log_softmax(output[idx_val], labels[idx_val])
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
