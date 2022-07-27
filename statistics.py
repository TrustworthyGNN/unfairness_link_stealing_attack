from gcn.utils import load_data_original, load_data_tu, load_dgl_fraud_data
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings

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

args = parser.parse_args()

# def calculating_stat(dataset):
# args.dataset = dataset
args.dataset = 'cora'
# args.dataset = 'citeseer'
# args.dataset = 'pubmed'
# args.dataset = 'COX2' #1
# args.dataset = 'DHFR' # 1
# args.dataset = 'ENZYMES' #1
# args.dataset = 'PROTEINS_full' #1

if args.dataset in ["citeseer", "cora", "pubmed"]:
    adj, features, labels, idx_train, idx_val, idx_test = load_data_original('./data/dataset/original/', args.dataset)
elif args.dataset in ['yelp', 'amazon']:
    adj, features, labels, idx_train, idx_val, idx_test = load_dgl_fraud_data(args.dataset)
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data_tu(args.dataset, args.dataset)
# adj, features, labels, idx_train, idx_val, idx_test = load_data_original('./data/dataset/original/', dataset)

labels = labels.tolist()
node_num=[]
label_number = max(labels)-min(labels)+1

adj = adj.to_dense().numpy()

degree = np.zeros(label_number)
degree_inner = np.zeros(label_number)
for i in range(len(labels)):
    d_index = labels[i]
    degree[d_index] += (np.count_nonzero(adj[i])-1)

    pos = np.nonzero(adj[i])[0].tolist()
    neigh_labels = np.array(labels)[pos].tolist()
    num_same_label = neigh_labels.count(d_index)-1
    degree_inner[d_index] += num_same_label

for i in range(label_number):
    node_num.append(labels.count(i))
# print(args.dataset, node_num)
ave_d = degree/node_num
ave_d_inner = degree_inner/node_num
print('ave degree ', ave_d)
print('ave degree inner ', ave_d_inner)

# plt.plot(node_num)
# plt.ylabel('node numbers')
# plt.show()

# est = ave_d/(ave_d-ave_d_inner)/node_num
est = ave_d/(ave_d-ave_d_inner)/node_num
print("est ", est)
fig, ax = plt.subplots()
ax.plot(est)
# plt.plot(np.ones_like(ave_d)/node_num)
# plt.ylabel('ave degree inner/all')
title = "statistic value"
ax.set_title(title)
plt.show()
