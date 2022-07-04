import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution, MLPReadout


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
