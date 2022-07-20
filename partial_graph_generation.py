import pickle as pkl
import json
import random
import time
import argparse
import numpy as np
from gcn import load_data, load_data_original

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cora', help='citeseer, cora or pubmed')
parser.add_argument('--filepath', type=str, default="data/cora/", help="data path")
parser.add_argument('--prediction_path', type=str, default='output/pred/', help='prediction saved path')
parser.add_argument('--saving_path', type=str, default='data/partial_graph_with_id/', help='partial graph saved path')

args = parser.parse_args()
dataset = args.dataset
file_path = args.filepath
prediction_path = args.prediction_path
saving_path = args.saving_path


def get_link(adj_mtx, num_of_node):
    unlink_list = []
    link_list = []
    existing_set = set([])
    rows, cols = adj_mtx.to_dense().nonzero(as_tuple=True)
    print("There are %d edges in this dataset" % len(rows))
    for row_idx in range(len(rows)):
        r_index = rows[row_idx].item()
        c_index = cols[row_idx].item()
        if r_index < c_index:
            # update link list
            link_list.append([r_index, c_index])
            existing_set.add(",".join([str(r_index), str(c_index)]))

    random.seed(1)
    start_at = time.time()
    while len(unlink_list) < len(link_list):
        if len(unlink_list) % 1000 == 0:
            print(len(unlink_list), time.time() - start_at)
        # get random row and col number
        random_row = random.randint(0, num_of_node - 1)
        random_col = random.randint(0, num_of_node - 1)
        # check if it needs replace
        if random_row > random_col:
            random_row, random_col = random_col, random_row
        edge_str = ",".join([str(random_row), str(random_col)])
        if (random_row != random_col) and (edge_str not in existing_set):
            # update unlink list
            unlink_list.append([random_row, random_col])
            existing_set.add(edge_str)
    return link_list, unlink_list


def generate_train_test(link_list, unlink_list, dense_pred, gcn_pred, train_ratio):
    train = []
    test = []

    train_len = len(link_list) * train_ratio
    for link_idx in range(len(link_list)):
        # print(i)
        link_id0 = link_list[link_idx][0]
        link_id1 = link_list[link_idx][1]

        line_link = {
            'label': 1,
            'gcn_pred0': gcn_pred[link_id0],
            'gcn_pred1': gcn_pred[link_id1],
            'gcn_pred0_label': gcn_pred[link_id0].index(max(gcn_pred[link_id0])),
            'gcn_pred1_label': gcn_pred[link_id1].index(max(gcn_pred[link_id1])),
            "dense_pred0": dense_pred[link_id0],
            "dense_pred1": dense_pred[link_id1],
            "feature_arr0": feature_arr[link_id0],
            "feature_arr1": feature_arr[link_id1],
            "id_pair": [int(link_id0), int(link_id1)]
        }

        unlink_id0 = unlink_list[link_idx][0]
        unlink_id1 = unlink_list[link_idx][1]

        line_unlink = {
            'label': 0,
            'gcn_pred0': gcn_pred[unlink_id0],
            'gcn_pred1': gcn_pred[unlink_id1],
            'gcn_pred0_label': gcn_pred[unlink_id0].index(max(gcn_pred[unlink_id0])),
            'gcn_pred1_label': gcn_pred[unlink_id1].index(max(gcn_pred[unlink_id1])),
            "dense_pred0": dense_pred[unlink_id0],
            "dense_pred1": dense_pred[unlink_id1],
            "feature_arr0": feature_arr[unlink_id0],
            "feature_arr1": feature_arr[unlink_id1],
            "id_pair": [int(unlink_id0), int(unlink_id1)]
        }

        if link_idx < train_len:
            train.append(line_link)
            train.append(line_unlink)
        else:
            test.append(line_link)
            test.append(line_unlink)

    with open(saving_path + "%s_train_ratio_%0.1f_train.json" % (dataset, train_ratio), "w") as wf1:
        for train_row in train:
            wf1.write("%s\n" % json.dumps(train_row))
        wf1.close()
    with open(saving_path + "%s_train_ratio_%0.1f_test.json" % (dataset, train_ratio), "w") as wf2:
        for test_row in test:
            wf2.write("%s\n" % json.dumps(test_row))
        wf2.close()


# load data
adj, features, labels, idx_train, idx_val, idx_test = load_data_original('./data/dataset/original/', dataset)
if isinstance(features, np.ndarray):
    feature_arr = features
else:
    feature_arr = features.numpy()
feature_arr = feature_arr.tolist()

# load saved model
dense_pred = pkl.loads(open(prediction_path + "%s_mlp_pred.pkl" % dataset, "rb").read())
gcn_pred = pkl.loads(open(prediction_path + "%s_gcn_pred.pkl" % dataset, "rb").read())
dense_pred = dense_pred.tolist()
gcn_pred = gcn_pred.tolist()

node_num = len(dense_pred)
link, unlink = get_link(adj, node_num)
random.shuffle(link)
random.shuffle(unlink)
label = []
for row in link:
    label.append(1)
for row in unlink:
    label.append(0)

# generate 10% to 100% of known edges
t_start = time.time()
for i in range(1, 11):
    print("generating: %d percent" % (i * 10), time.time() - t_start)
    generate_train_test(link, unlink, dense_pred, gcn_pred, i / 10.0)
