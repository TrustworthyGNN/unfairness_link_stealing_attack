import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution, MLPReadout, GraphAttentionLayer, SageConvLayer


class GCN(nn.Module):
    def __init__(self, feat, hidden_dim, n_classes, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(feat, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class MLPNet(nn.Module):
    def __init__(self, feat, hidden_dim, n_classes, dropout):
        super(MLPNet, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(feat, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act_fn = torch.nn.functional.logsigmoid
        self._init_weights()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x, adj):
        # input embedding
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GraphSage(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphSage, self).__init__()
        self.sage1 = SageConvLayer(nfeat, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)


    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        return x