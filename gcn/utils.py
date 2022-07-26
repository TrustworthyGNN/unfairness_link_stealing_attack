import os
import pandas as pd
import dgl
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import torch
from dgl.data import FraudDataset
from scipy.sparse.linalg import eigsh
import sys
from scipy.spatial import distance
from scipy.spatial import distance_matrix


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    return np.array(idx_map)


def kl_divergence(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon
    divergence = np.sum(P * np.log(P / Q))
    return divergence


def js_divergence(P, Q):
    return distance.jensenshannon(P, Q, 2.0)


def entropy(P):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    entropy_value = -np.sum(P * np.log(P))
    return entropy_value


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data_original(datapath_str, dataset_str):
    """
    Loads input data from gcn/data/dataset directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(datapath_str, dataset_str, names[i]),
                  'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(
        datapath_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder),
            max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[
                                    test_idx_range, :]  # order the test features
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # normalize adjacency matrix
    adj = normalize(adj + sp.eye(adj.shape[0]))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[
                                  test_idx_range, :]  # order the test labels

    all_id_list = list(range(len(labels)))
    train_ratio = 0.1
    idx_train = all_id_list[:int(len(all_id_list) * train_ratio)]
    idx_val = all_id_list[int(len(all_id_list) * train_ratio):]
    idx_test = all_id_list
    features = torch.FloatTensor(np.array(features.todense()))
    # add idx filter, since all zeros exist in the label matrix
    labels = torch.LongTensor(np.where(labels)[1])
    if labels.shape[0] != features.shape[0]:
        idx_filter = np.where(labels)[0].tolist()
        idx_train = list(set(idx_filter) & set(idx_train))
        idx_val = list(set(idx_filter) & set(idx_val))
        idx_test = list(set(idx_filter) & set(idx_test))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_tu(datapath_str, dataset_str):
    print("load_data_tu: %s" % dataset_str)
    names = ['attr', 'graph']
    objects = []
    for i in range(len(names)):
        with open(
                "data/dataset/tu/DS_all/{}/{}_{}.pkl".format(dataset_str, dataset_str, names[i]),
                'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    attr, graph = tuple(objects)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # normalize adjacency matrix
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = np.array([attr[i]["feature_vec"] for i in range(len(attr))])
    labels = np.array([attr[i]["label"] for i in range(len(attr))])

    all_id_list = list(range(len(labels)))
    train_ratio = 0.1
    idx_train = all_id_list[:int(len(all_id_list) * train_ratio)]
    idx_val = all_id_list[int(len(all_id_list) * train_ratio):]
    idx_test = all_id_list
    features = torch.FloatTensor(features)
    # add idx filter, since all zeros exist in the label matrix
    labels = torch.LongTensor(np.where(labels)[1])
    if labels.shape[0] != features.shape[0]:
        idx_filter = np.where(labels)[0].tolist()
        idx_train = list(set(idx_filter) & set(idx_train))
        idx_val = list(set(idx_filter) & set(idx_val))
        idx_test = list(set(idx_filter) & set(idx_test))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, idx_train, idx_val, idx_test


def load_nifty(dataset, path="./data/dataset/"):
    path = path + dataset
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    if dataset == 'credit':
        predict_attr = "NoDefaultNextMonth"
        header.remove("NoDefaultNextMonth")
        header.remove('Single')
    else:
        predict_attr = "GoodCustomer"
        header.remove("GoodCustomer")
        header.remove('OtherLoansAtStore')
        header.remove('PurposeOfLoan')
        # Sensitive Attribute
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
        idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    # remove outline
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # normalize adjacency matrix
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    all_id_list = list(range(len(labels)))
    train_ratio = 0.1
    idx_train = all_id_list[:int(len(all_id_list) * train_ratio)]
    idx_val = all_id_list[int(len(all_id_list) * train_ratio):]
    idx_test = all_id_list

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels, idx_train, idx_val, idx_test


def load_dgl_fraud_data(name='yelp'):
    dataset = FraudDataset(name)
    graph = dataset[0]
    features = graph.ndata['feature']
    labels = graph.ndata['label']
    adj = dgl.to_homogeneous(graph).adj()
    all_id_list = list(range(len(labels)))
    # method 1: original split provided by dgl
    # idx_train = np.where(graph.ndata['train_mask'])[0].tolist()
    # idx_val = np.where(graph.ndata['val_mask'])[0].tolist()
    # idx_test = np.where(graph.ndata['test_mask'])[0].tolist()

    # method 2: split ratio used in "stealing link" paper
    train_ratio = 0.1
    idx_train = all_id_list[:int(len(all_id_list) * train_ratio)]
    idx_val = all_id_list[int(len(all_id_list) * train_ratio):]
    idx_test = all_id_list

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        features = sp.csr_matrix(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update(
        {placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(
        adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


if __name__ == '__main__':
    datapath_str = "data/dataset/original/"
    dataset_str = "citeseer"
    load_data(datapath_str, dataset_str)
