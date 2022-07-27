import time
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev, \
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import json
import argparse
import numpy as np
import copy

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

import matplotlib
import matplotlib.pyplot as plt

from gcn.utils import entropy, kl_divergence, js_divergence

from scipy.stats import wasserstein_distance

# from fairness import display_fairness
from util.fairness import display_fairness

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='cora', help='citeseer, cora or pubmed')
parser.add_argument('--datapath', type=str, default="data/dataset/original/", help="data path")
parser.add_argument('--prediction_path', type=str, default='data/pred/', help='prediction saved path')
parser.add_argument('--partial_graph_path', type=str, default='data/partial_graph_with_id/',
                    help='partial graph saved path')
parser.add_argument('--ratio', type=str, default='0.5', help='(0.1,1.0,0.1)')

args = parser.parse_args()

args.dataset = 'cora'
# args.dataset = 'citeseer'
# args.dataset = 'pubmed'
# args.dataset = 'COX2' #1
# args.dataset = 'DHFR' # 1
# args.dataset = 'ENZYMES' #1
# args.dataset = 'PROTEINS_full' #1
# args.dataset = 'credit'
# args.dataset = 'german'
print('dataset is ', args.dataset)

dataset = args.dataset
datapath = args.datapath
prediction_path = args.prediction_path
partial_graph_path = args.partial_graph_path
ratio = args.ratio

# using_broad_threshold=False


def plot_g_auc(g_auc_dict, label_num, desc="similarity score"):
    # if dataset=='cora':
    #     label_num =7
    # else:
    #     assert False
    class_in_prediction = []
    for i in range(label_num):
        for key in g_auc_dict:
            if str(i) in key:
                class_in_prediction.append(i)
                break

    # class0 = ['class-'+str(i) for i in range(label_num)]
    # class1 = ['class-'+str(i) for i in range(label_num)]
    class0 = ['class-' + str(i) for i in class_in_prediction]
    class1 = ['class-' + str(i) for i in class_in_prediction]

    # class0 = ["cucumber", "tomato", "lettuce", "asparagus",
    #               "potato", "wheat", "barley"]
    # class1 = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    harvest = np.ones([len(class_in_prediction), len(class_in_prediction)]) * (-1)
    # for key in g_auc_dict:
    #     harvest[int(key[0]),int(key[1])]=round(g_auc_dict[key]*100,2)
    #     if key[0]!=key[1]:
    #         harvest[int(key[1]), int(key[0])] = round(g_auc_dict[key]*100, 2)
    for key in g_auc_dict:
        harvest[class_in_prediction.index(int(key[0])), class_in_prediction.index(int(key[1]))] = round(
            g_auc_dict[key] * 100, 2)
        if key[0] != key[1]:
            harvest[class_in_prediction.index(int(key[1])), class_in_prediction.index(int(key[0]))] = round(
                g_auc_dict[key] * 100, 2)
    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)
    fig.colorbar(im, ax=ax, )

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(class1)), labels=class1)
    ax.set_yticks(np.arange(len(class0)), labels=class0)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class0)):
        for j in range(len(class1)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    title = 'AUC based on ' + desc + ' distance'
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig('result/' + args.dataset + ' ' + title + '.png', dpi=600)
    plt.show()

def plot_g_auc_curve(g_auc_dict, label_num, group_dict, desc = 'distance'):
    inter=[]
    intra=[]
    label_intra=[]
    label_inter = []
    num_intra = []
    num_inter = []
    for i in range(label_num):
        # if g_auc_dict.has_key(str(i)+str(i)):
        if (str(i)+str(i)) in g_auc_dict.keys():
            intra.append(g_auc_dict[str(i)+str(i)])
            label_intra.append(i)
            num_intra.append(len(group_dict[str(i)+str(i)]))
        # if g_auc_dict.has_key(str(i)+'-'):
        if (str(i)+'-') in g_auc_dict.keys():
            inter.append(g_auc_dict[str(i) + '-'])
            label_inter.append(i)
            num_inter.append(len(group_dict[str(i)+'-']))

    x_intra = label_intra
    y_intra_auc = intra
    y_intra_num = np.array(num_intra)/max(num_intra)
    x_inter = label_inter
    y_inter_auc = inter
    y_inter_num = np.array(num_inter)/max(num_inter)

    fig, ax = plt.subplots()
    # Using set_dashes() to modify dashing of an existing line
    line1, = ax.plot(x_intra, y_intra_auc, label='Intra-class AUC')
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

    # Using plot(..., dashes=...) to set the dashing when creating a line
    line2, = ax.plot(x_inter, y_inter_auc, dashes=[6, 2], label='Inter-class AUC')

    # line3, = ax.plot(x_intra, y_intra_num, label='Intra-class num')
    # line3.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    #
    # line4, = ax.plot(x_inter, y_inter_num, dashes=[6, 2], label='Inter-class num')

    title = desc
    ax.set_title(title)

    ax.legend()
    plt.show()

    return True

def plot_g_auc_curve_onenode(g_auc_dict, label_num, desc = 'distance'):
    auc = []
    label = []
    for i in range(label_num):
        if (str(i)) in g_auc_dict.keys():
            auc.append(g_auc_dict[str(i)])
            label.append(i)

    x = label
    y = auc

    fig, ax = plt.subplots()
    # Using set_dashes() to modify dashing of an existing line
    line1, = ax.plot(x, y, label='AUC')
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    # est = calculating_stat(args.dataset)
    # # # Using plot(..., dashes=...) to set the dashing when creating a line
    # line2, = ax.plot(x, est, dashes=[6, 2], label='statistics')
    #
    # # line3, = ax.plot(x_intra, y_intra_num, label='Intra-class num')
    # # line3.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    # #
    # line4, = ax.plot(x_inter, y_inter_num, dashes=[6, 2], label='Inter-class num')

    title = desc
    ax.set_title(title)

    ax.legend()
    plt.savefig('result/' + args.dataset + ' empirical study link stealing '+desc+'.png', dpi=600)
    plt.show()

    return True


def evaluation(label, pred_label):
    acc = accuracy_score(y_true=label, y_pred=pred_label)
    prec = precision_score(y_true=label, y_pred=pred_label)
    reca = recall_score(y_true=label, y_pred=pred_label)
    f1 = f1_score(y_true=label, y_pred=pred_label)
    return acc, prec, reca, f1


def attack_0(target_posterior_list):
    sim_metric_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    sim_list_target = [[] for _ in range(len(sim_metric_list))]
    for i in range(len(target_posterior_list)):
        for j in range(len(sim_metric_list)):
            # using target only
            target_sim = sim_metric_list[j](target_posterior_list[i][0],
                                            target_posterior_list[i][1])
            sim_list_target[j].append(target_sim)
    return sim_list_target


def attack_0_entropy(target_posterior_list):
    sim_metric_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    sim_list_target = [[] for _ in range(len(sim_metric_list))]
    for i in range(len(target_posterior_list)):
        for j in range(len(sim_metric_list)):
            # using target only

            # entropy_n0 = entropy(np.array(target_posterior_list[i][0]))
            # entropy_n1 = entropy(np.array(target_posterior_list[i][1]))
            # target_sim = sim_metric_list[j](entropy_n0, entropy_n1)

            kl_01 = kl_divergence(np.array(target_posterior_list[i][0]), np.array(target_posterior_list[i][1]))
            kl_10 = kl_divergence(np.array(target_posterior_list[i][1]), np.array(target_posterior_list[i][0]))
            target_sim = (kl_01 + kl_10) / 2

            # target_sim = js_divergence(np.array(target_posterior_list[i][0]), np.array(target_posterior_list[i][1]))

            # target_sim = wasserstein_distance(np.array(target_posterior_list[i][0]), np.array(target_posterior_list[i][1]))

            sim_list_target[j].append(target_sim)
    return sim_list_target


def write_auc(pred_prob_list, label, desc):
    print("Attack 0 " + desc)
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    timestamp = str(round(time.time()))
    with open("result/attack_0_auc_at_%s.txt" % timestamp, "a") as wf:
        for i in range(len(sim_list_str)):
            pred = np.array(pred_prob_list[i], dtype=np.float64)
            where_are_nan = np.isnan(pred)
            where_are_inf = np.isinf(pred)
            pred[where_are_nan] = 0
            pred[where_are_inf] = 0

            i_auc = roc_auc_score(label, pred)

            # pred_label = [1 if p >= 0.5 else 0 for p in pred]

            if i_auc < 0.5:
                i_auc = 1 - i_auc
            print(sim_list_str[i], i_auc)
            wf.write(
                "%s,%s,%d,%0.5f,%s\n" %
                (dataset, "attack0_%s_%s" %
                 (desc, sim_list_str[i]), -1, i_auc, ratio))

            # show the fairness metrics using fairlearn
            # display_fairness(label, pred_label, pd.Series(label))
    wf.close()


def group_processing(group_dict, label):
    # in case some keys have single label, this function will combine them with a key with different label
    sub_group_dict = {}
    hybird_key = ''
    hybird_index = []
    # key_with_largest_size=None
    # largest_size
    for key in group_dict:
        g_label = label[group_dict[key]]
        if max(g_label) - min(g_label) == 0:
            sub_group_dict[key] = group_dict[key]
            hybird_key += ',' + key
            hybird_index += group_dict[key]
    for key in sub_group_dict:
        group_dict.pop(key)
    if len(hybird_index) > 0:
        hybird_label = label[hybird_index]
        select_key = None
        # select_size = 0
        select_size = float('inf')
        if max(hybird_label) - min(hybird_label) == 0:
            for key in group_dict:
                if len(group_dict[key]) < select_size:
                    select_size = len(group_dict[key])
                    select_key = key
            hybird_key = select_key + hybird_key
            hybird_index = group_dict[select_key] + hybird_index
        else:
            group_dict[hybird_key[1:]] = hybird_index

        group_dict.pop(select_key)
        group_dict[hybird_key] = hybird_index
    return group_dict

def grouping_onenode(group_dict, node_label_num): # grouping edges concerning one node, the output has node_label_num groups
    group_dict_new={}
    for i in range(node_label_num):
        tmp_index = []
        for key in group_dict:
            if str(i) in key:
                tmp_index+=group_dict[key]
        if len(tmp_index)>0:
            group_dict_new[str(i)]=tmp_index
    return group_dict_new


def grouping_inter_intra(group_dict):  # grouping edges concerning one node, the output has 2 groups
    group_dict_new = {}
    group_dict_new['intra'] = []
    group_dict_new['inter'] = []
    for key in group_dict:
        if key[0] == key[1]:
            group_dict_new['intra'] = group_dict_new['intra'] + group_dict[key]
        else:
            group_dict_new['inter'] = group_dict_new['inter'] + group_dict[key]
    return group_dict_new

def grouping_onenode_inter_intra(group_dict, node_label_num): # grouping edges concerning one node, the output has 2* node_label_num groups
    group_dict_new={}
    for i in range(node_label_num):
        group_dict_new[str(i) + str(i)] = []
        group_dict_new[str(i) + '-'] = []
        for key in group_dict:
            if key.count(str(i)) == 1:
                group_dict_new[str(i) + '-'] = group_dict_new[str(i) + '-'] + group_dict[key]
            if key.count(str(i)) == 2:
                group_dict_new[str(i) + str(i)] = group_dict_new[str(i) + str(i)] + group_dict[key]
    nullkeys = []
    for key in group_dict_new:
        if len(group_dict_new[key]) == 0:
            nullkeys.append(key)
    for key in nullkeys:
        group_dict_new.pop(key)
    return group_dict_new

def grouping_processing_broad(group_dict): # the key with key[0]!=key[1] will be updated by combining key[0]key[1], key[0]key[0],key[1]key[1]
    group_dict_broad = {}
    for key in group_dict:
        if key[0] == key[1]:
            group_dict_broad[key] = group_dict[key]
        else:
            group_dict_broad[key] = group_dict[key] + group_dict[key[0] + key[0]] + group_dict[key[1] + key[1]]
    return group_dict_broad, group_dict


def grouping_processing_broad_hybird(group_dict):
    # the key with key[0]!=key[1] will be updated by combining key[0]key[1], key[0]key[0],key[1]key[1]
    # the keys with key[0]==key[1] will be merged as intra
    group_dict_broad = {}
    group_dict_broad['intra'] = []
    group_dict_new = {}
    group_dict_new['intra'] = []
    for key in group_dict:
        if key[0] == key[1]:
            group_dict_broad['intra'] = group_dict_broad['intra'] + group_dict[key]
            group_dict_new['intra'] = group_dict_new['intra'] + group_dict[key]
        else:
            group_dict_broad[key] = group_dict[key] + group_dict[key[0] + key[0]] + group_dict[key[1] + key[1]]
            group_dict_new[key] = group_dict[key]
    return group_dict_broad, group_dict_new


def grouping_processing_broad_hybird_v2(group_dict):
    # the key with key[0]!=key[1] will be calculated by them self
    # the keys with key[0]==key[1] will be merged as intra
    group_dict_broad = {}
    group_dict_broad['intra'] = []
    group_dict_new = {}
    group_dict_new['intra'] = []
    for key in group_dict:
        if key[0] == key[1]:
            group_dict_broad['intra'] = group_dict_broad['intra'] + group_dict[key]
            group_dict_new['intra'] = group_dict_new['intra'] + group_dict[key]
        else:
            group_dict_broad[key] = group_dict[key]
            group_dict_new[key] = group_dict[key]
    return group_dict_broad, group_dict_new


def grouping_inter_intra_hybird(group_dict):
    # the inter-class edge will be merged with intra-class edges, while intra-class edges will keep an isolated group
    group_dict_broad = {}
    group_dict_broad['intra'] = []
    group_dict_broad['inter'] = []
    group_dict_new = {}
    group_dict_new['intra'] = []
    group_dict_new['inter'] = []
    for key in group_dict:
        group_dict_broad['inter'] = group_dict_broad['inter'] + group_dict[key]
        if key[0] == key[1]:
            group_dict_broad['intra'] = group_dict_broad['intra'] + group_dict[key]
            group_dict_new['intra'] = group_dict_new['intra'] + group_dict[key]
        else:
            group_dict_new['inter'] = group_dict_new['inter'] + group_dict[key]
    return group_dict_broad, group_dict_new


def write_auc_group(pred_prob_list, label, group_dict, node_label_num, desc):
    print("Attack 0 " + desc)
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    timestamp = str(round(time.time()))

    label = np.array(label)

    # group_dict = group_processing(group_dict, label)
    # group_dict = grouping_onenode(group_dict, node_label_num)
    group_dict = grouping_inter_intra(group_dict)
    # group_dict = grouping_onenode_inter_intra(group_dict, node_label_num)
    with open("result/attack_0_auc_group_at_%s.txt" % timestamp, "a") as wf:

        # for key in group_dict:
        #     group_auc[key] = -1
        #     group_size[key] = len(group_dict[key])

        for i in range(len(sim_list_str)):
            group_auc = []
            group_auc_set = {}
            group_size = []
            group_size_set = {}

            pred = np.array(pred_prob_list[i], dtype=np.float64)
            where_are_nan = np.isnan(pred)
            where_are_inf = np.isinf(pred)
            pred[where_are_nan] = 0
            pred[where_are_inf] = 0

            for key in group_dict:
                g_label = label[group_dict[key]]
                g_pred = pred[group_dict[key]]
                g_auc = roc_auc_score(g_label, g_pred)
                if g_auc < 0.5:
                    g_auc = 1 - g_auc
                print(sim_list_str[i], key, round(g_auc * 100, 2))
                wf.write(
                    "%s,%s,%s,%0.5f\n" %
                    (dataset, "attack0_%s_%s" %
                     (desc, sim_list_str[i]), key, g_auc))
                group_auc.append(g_auc)
                group_auc_set[key] = g_auc
                group_size.append(len(group_dict[key]))
                group_size_set[key] = len(group_dict[key])
            group_auc = np.array(group_auc)
            group_size = np.array(group_size)
            group_based_ave_auc = np.inner(group_auc, group_size) / group_size.sum()

            with_plot = True
            with_plot_curve = False
            with_plot_curve_onenode = False
            for key in group_dict:
                if key == 'inter' or key == 'intra':
                    with_plot = False
                    break
                if len(key) == 1:
                    with_plot = False
                    with_plot_curve_onenode = True
                    break
                if '-' in key:
                    with_plot = False
                    with_plot_curve = True
                    break
            if with_plot:
                plot_g_auc(group_auc_set, node_label_num, desc=sim_list_str[i])
            if with_plot_curve:
                plot_g_auc_curve(group_auc_set, node_label_num, group_dict, desc=sim_list_str[i]+' distance')
            if with_plot_curve_onenode:
                plot_g_auc_curve_onenode(group_auc_set, node_label_num, desc=sim_list_str[i]+' distance')
            # group_based_ave_auc = np.array(group_auc).sum() / len(group_auc)
            # print(sim_list_str[i]+" group-based-ave-auc "+ str(group_based_ave_auc))
            # wf.write(
            #     "%s,%s,%0.5f\n" %
            #     (dataset, "attack0_%s_%s" %
            #      ("******** group-based-ave-auc ", sim_list_str[i]), group_based_ave_auc))
            # show the fairness metrics using fairlearn
            # display_fairness(label, pred_label, pd.Series(label))
    wf.close()


def write_acc(pred_prob_list, label):
    print("Attack 0 " + "Kmeans")
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    timestamp = str(round(time.time()))
    with open("result/attack_0_acc_at_%s.txt" % timestamp, "a") as wf:
        for i in range(len(sim_list_str)):
            pred = np.array(pred_prob_list[i], dtype=np.float64)
            pred = np.ones_like(pred) - pred
            where_are_nan = np.isnan(pred)
            where_are_inf = np.isinf(pred)
            pred[where_are_nan] = 0
            pred[where_are_inf] = 0

            kmeans = KMeans(n_clusters=2, random_state=0).fit(pred.reshape(-1, 1))
            threshold = (kmeans.cluster_centers_[0] + kmeans.cluster_centers_[1]) / 2
            pred_label = [1 if p >= threshold else 0 for p in pred]
            # pred_label = kmeans.labels_
            # score1 = np.inner(pred, pred_label)
            # score0 = np.inner(pred, np.ones_like(pred_label)-pred_label)
            # if score1< score0:
            #     pred_label = np.ones_like(pred_label)-pred_label
            acc, prec, reca, f1 = evaluation(label, pred_label)

            # precision_score(y_true=label, y_pred=pred_label)
            print(sim_list_str[i], 'acc: ', round(acc * 100, 2), 'f1 ', f1)
            wf.write(
                "%s,%s,%s,%0.5f,%s,%0.5f\n" %
                (dataset, "attack0_%s" %
                 (sim_list_str[i]), 'acc: ', acc, 'f1: ', f1))

            # show the fairness metrics using fairlearn
            # display_fairness(label, pred_label, pd.Series(label))
    wf.close()


def write_acc_group(pred_prob_list, label, group_dict, node_label_num, desc):
    print("Attack 0 " + desc)
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    timestamp = str(round(time.time()))

    label = np.array(label)

    # if using_broad_threshold:
    #     # assert False
    #     # group_dict_broad, group_dict = grouping_processing_broad(group_dict)
    #     # group_dict_broad, group_dict = grouping_processing_broad_hybird(group_dict)
    #     # group_dict_broad, group_dict = grouping_inter_intra_hybird(group_dict)
    #     group_dict_broad, group_dict = grouping_processing_broad_hybird_v2(group_dict)
    # else:
    # group_dict = group_processing(group_dict, label)
    # group_dict = grouping_onenode(group_dict, node_label_num)
    group_dict = grouping_inter_intra(group_dict)
    # group_dict = grouping_onenode_inter_intra(group_dict, node_label_num)
    # assert False
    with open("result/attack_0_evaluation_group_at_%s.txt" % timestamp, "a") as wf:

        for i in range(len(sim_list_str)):
            acc_group_threshold = []
            f1_group_threshold = []
            reca_group_threshold = []
            prec_group_threshold = []
            group_size = []

            acc_single_threshold = []
            f1_single_threshold = []
            reca_single_threshold = []
            prec_single_threshold = []

            pred = np.array(pred_prob_list[i], dtype=np.float64)
            pred = np.ones_like(pred) - pred
            where_are_nan = np.isnan(pred)
            where_are_inf = np.isinf(pred)
            pred[where_are_nan] = 0
            pred[where_are_inf] = 0

            kmeans = KMeans(n_clusters=2, random_state=0).fit(pred.reshape(-1, 1))
            threshold_single = (kmeans.cluster_centers_[0] + kmeans.cluster_centers_[1]) / 2
            pred_label_single_threshold = np.array([1 if p >= threshold_single else 0 for p in pred])

            for key in group_dict:
                g_label = label[group_dict[key]]  # ground truth label
                # if using_broad_threshold:
                #     g_pred = pred[group_dict_broad[key]]
                # else:
                g_pred = pred[group_dict[key]] # predict score

                g_pred_label_single_threshold = pred_label_single_threshold[group_dict[key]]

                kmeans = KMeans(n_clusters=2, random_state=0).fit(g_pred.reshape(-1, 1))
                threshold_group = (kmeans.cluster_centers_[0] + kmeans.cluster_centers_[1]) / 2
                # if using_broad_threshold:
                #     g_pred = pred[group_dict[key]]
                #     g_pred_label_group_threshold = [1 if p >= threshold_group else 0 for p in g_pred]
                # else:
                g_pred_label_group_threshold = [1 if p >= threshold_group else 0 for p in g_pred]

                g_acc_s, g_prec_s, g_reca_s, g_f1_s = evaluation(g_label, g_pred_label_single_threshold)
                g_acc_g, g_prec_g, g_reca_g, g_f1_g = evaluation(g_label, g_pred_label_group_threshold)

                # print(sim_list_str[i], key, 'acc_s: ', g_acc_s, 'acc_g: ', g_acc_g, 'prec_s: ', g_prec_s, 'prec_g ', g_prec_g,
                #                           'reca_s: ', g_reca_s, 'reca_g: ', g_reca_g, 'f1_s: ', g_f1_s, 'f1_g: ', g_f1_g)
                # wf.write(
                #     "%s,%s,%s,%s,%0.5f,%s,%0.5f,%s,%0.5f,%s,%0.5f,%s,%0.5f,%s,%0.5f,%s,%0.5f,%s,%0.5f\n" %
                #     (dataset, "attack0_%s" % (sim_list_str[i]), key,
                #      'acc_s: ', g_acc_s, 'acc_g: ', g_acc_g, 'prec_s: ', g_prec_s, 'prec_g ', g_prec_g,
                #      'reca_s: ', g_reca_s, 'reca_g: ', g_reca_g, 'f1_s: ', g_f1_s, 'f1_g: ', g_f1_g))
                # print(sim_list_str[i], key, 'acc_s: ', g_acc_s, 'acc_g: ', g_acc_g, 'reca_s: ', g_reca_s, 'reca_g: ', g_reca_g)
                # wf.write(
                #     "%s,%s,%s,%s,%0.5f,%s,%0.5f,%s,%0.5f,%s,%0.5f\n" %
                #     (dataset, "attack0_%s" % (sim_list_str[i]), key,
                #      'acc_s: ', g_acc_s, 'acc_g: ', g_acc_g, 'reca_s: ', g_reca_s, 'reca_g: ', g_reca_g))
                print(sim_list_str[i], key, 'group_size: ', len(group_dict[key]), 'acc_s: ', round(g_acc_s * 100, 2),
                      'acc_g: ', round(g_acc_g * 100, 2))
                wf.write(
                    "%s,%s,%s,%s,%0.5f,%s,%0.5f,%s,%0.5f\n" %
                    (dataset, "attack0_%s" % (sim_list_str[i]), key, 'group_size: ', len(group_dict[key]),
                     'acc_s: ', g_acc_s, 'acc_g: ', g_acc_g))

                acc_single_threshold.append(g_acc_s)
                f1_single_threshold.append(g_f1_s)
                reca_single_threshold.append(g_reca_s)
                prec_single_threshold.append(g_prec_s)

                acc_group_threshold.append(g_acc_g)
                f1_group_threshold.append(g_f1_g)
                reca_group_threshold.append(g_reca_g)
                prec_group_threshold.append(g_prec_g)

                group_size.append(len(group_dict[key]))
            acc_group_threshold = np.array(acc_group_threshold)
            acc_single_threshold = np.array(acc_single_threshold)
            # group_f1 = np.array(group_f1)
            group_size = np.array(group_size)
            group_based_ave_acc = np.inner(acc_group_threshold, group_size) / group_size.sum()
            single_based_ave_acc = np.inner(acc_single_threshold, group_size) / group_size.sum()
            print(sim_list_str[i] + " group-based-ave-acc ", round(group_based_ave_acc * 100, 2),
                  " single-based-ave-acc ", round(single_based_ave_acc * 100, 2))
            # group_based_ave_f1 = np.inner(group_f1, group_size) / group_size.sum()
            # print(sim_list_str[i]+" group-based-ave-acc "+ str(group_based_ave_acc)+" group_based_ave_f1 "+ str(group_based_ave_f1))
            # wf.write(
            #     "%s,%s,%0.5f\n" %
            #     (dataset, "attack0_%s_%s" %
            #      ("******** group-based-ave-acc ", sim_list_str[i]), group_based_ave_acc))

            # show the fairness metrics using fairlearn
            # display_fairness(label, pred_label, pd.Series(label))
    wf.close()


def viz(pred_prob_list, label, group_dict, node_label_num):
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    timestamp = str(round(time.time()))

    label = np.array(label)

    # group_dict = group_processing(group_dict, label)
    group_dict = grouping_inter_intra(group_dict)
    # group_dict = grouping_onenode_inter_intra(group_dict, node_label_num)

    ave_improvement = 0
    ave_improvement_inter = 0
    for i in range(len(sim_list_str)):
        acc_group_threshold = []
        f1_group_threshold = []
        reca_group_threshold = []
        prec_group_threshold = []
        group_size = []

        acc_single_threshold = []
        f1_single_threshold = []
        reca_single_threshold = []
        prec_single_threshold = []

        pred = np.array(pred_prob_list[i], dtype=np.float64)
        pred = np.ones_like(pred) - pred
        # pred = pow(pred,200)
        where_are_nan = np.isnan(pred)
        where_are_inf = np.isinf(pred)
        pred[where_are_nan] = 0
        pred[where_are_inf] = 0

        kmeans = KMeans(n_clusters=2, random_state=0).fit(pred.reshape(-1, 1))
        threshold_single = (kmeans.cluster_centers_[0] + kmeans.cluster_centers_[1]) / 2
        pred_label_single_threshold = np.array([1 if p >= threshold_single else 0 for p in pred])

        index_1 = [i for i in range(len(label)) if label[i] == 1]
        index_0 = [i for i in range(len(label)) if label[i] == 0]

        n_bins = 100
        colors = ['red', 'blue']
        colors_label = ["edge ground truth: 1", 'edge ground truth: 0']
        distance_1 = pred[index_1]
        distance_0 = pred[index_0]
        # if key == 'intra':
        #     distance_1 = pow(distance_1, 200)
        #     distance_0 = pow(distance_0, 200)
        x_multi = [distance_1, distance_0]
        fig, axs = plt.subplots(nrows=1, ncols=1)
        # axs.hist(distance_1, n_bins = 50, density=True, histtype='bar', rwidth=0.5)
        # axs.hist(distance_0, n_bins = 50, density=True, histtype='bar', rwidth=0.3)
        axs.hist(x_multi, n_bins, histtype='bar', rwidth=0.8, color=colors, label=colors_label)
        axs.legend(prop={'size': 10})
        title = 'overall distribution with ' + sim_list_str[i] + "\n thresh_single-" + str(
            round(threshold_single.item(), 2))
        axs.set_title(dataset + ' '+title)

        fig.tight_layout()
        plt.savefig('result/'+ dataset + ' overall ' + sim_list_str[i]+'.png', dpi=600)
        plt.show()

        inter_improve = 0
        for key in group_dict:
            g_label = label[group_dict[key]]  # ground truth label

            g_index_1 = [group_dict[key][i] for i in range(len(group_dict[key])) if label[group_dict[key][i]] == 1]
            g_index_0 = [group_dict[key][i] for i in range(len(group_dict[key])) if label[group_dict[key][i]] == 0]

            g_pred = pred[group_dict[key]]  # predict score

            # if key == 'intra':
            #     g_pred = pow(g_pred,2000)

            g_pred_label_single_threshold = pred_label_single_threshold[group_dict[key]]

            # kmeans = KMeans(n_clusters=2, random_state=0).fit(g_pred.reshape(-1, 1))
            # threshold_group = (kmeans.cluster_centers_[0] + kmeans.cluster_centers_[1]) / 2
            # comp = 4
            # gm = GaussianMixture(n_components=comp, random_state=0).fit(g_pred.reshape(-1, 1))
            # g_means = gm.means_.reshape(1,-1).tolist()[0]
            # threshold_group = (g_means[comp-2] + g_means[comp-1]) / 2
            # threshold_group = np.array(threshold_group)

            if key == 'intra':
                threshold_group = threshold_single
            #     ratio = len(group_dict[key])/label.shape[0]
            #     # ratio = 1- len(group_dict[key]) / label.shape[0]
            #     # ratio = 0.1
            #     threshold_group = ( ratio * threshold_group + (1-ratio) * threshold_single)/1
            if key == 'inter':
            #     # g_pred = pow(g_pred, 5)
            #     # threshold_group = np.array(1)
            #     # ratio = len(group_dict[key]) / label.shape[0]
            #     # ratio = 1 - len(group_dict[key]) / label.shape[0]
            #     # ratio = 0.8
            #     # threshold_group = ( ratio * threshold_group + (1-ratio) * threshold_single)/1
            #     threshold_group = ((1- threshold_group) +  threshold_single) / 2
            #     threshold_group *= len(group_dict['intra'])/len(group_dict['inter'])
            #     threshold_group = min(np.array(1), threshold_single*len(group_dict['intra'])/len(group_dict['inter']))
            #     threshold_group = max(threshold_single* len(group_dict['inter'])/len(group_dict['intra']), min(1,threshold_group))
                ratio = len(group_dict[key]) / label.shape[0]
                threshold_group = ratio * threshold_single + (1-ratio)*1
                # threshold_group = (1 - ratio) * threshold_single + ratio * 1

            # if using_broad_threshold:
            #     g_pred = pred[group_dict[key]]
            #     g_pred_label_group_threshold = [1 if p >= threshold_group else 0 for p in g_pred]
            # else:
            g_pred_label_group_threshold = [1 if p >= threshold_group else 0 for p in g_pred]

            g_acc_s, g_prec_s, g_reca_s, g_f1_s = evaluation(g_label, g_pred_label_single_threshold)
            g_acc_g, g_prec_g, g_reca_g, g_f1_g = evaluation(g_label, g_pred_label_group_threshold)
            print(sim_list_str[i], key, 'size: ', len(group_dict[key]), 'acc_s: ', round(g_acc_s*100,2), 'acc_g: ', round(g_acc_g*100,2))

            # if i >= 0 and i < 4:
            if i >= 0:
                n_bins = 100
                colors = ['red', 'blue']
                colors_label = ["ground truth: edge = 1", 'ground truth: edge = 0']
                distance_1 = pred[g_index_1]
                distance_0 = pred[g_index_0]
                # if key == 'intra':
                #     distance_1 = pow(distance_1, 200)
                #     distance_0 = pow(distance_0, 200)
                x_multi = [distance_1, distance_0]
                fig, axs = plt.subplots(nrows=1, ncols=1)
                # axs.hist(distance_1, n_bins = 50, density=True, histtype='bar', rwidth=0.5)
                # axs.hist(distance_0, n_bins = 50, density=True, histtype='bar', rwidth=0.3)
                axs.hist(x_multi, n_bins, histtype='bar', rwidth=0.8, color=colors, label=colors_label)
                axs.legend(prop={'size': 10})
                # title = " thresh_single-" + str(round(threshold_single.item(),2)) + ", thresh_group-" + str(round(threshold_group.item(),2))
                # axs.set_title(sim_list_str[i]+' '+key+' '+title)

                title = key+'-class distribution with ' + sim_list_str[i] + "\n thresh_single-" + str(round(threshold_single.item(),2)) + ", thresh_group-" + str(round(threshold_group.item(),2))
                axs.set_title(dataset + ' ' + title)

                fig.tight_layout()
                plt.savefig('result/' + args.dataset + ' '+key+'-class with ' + sim_list_str[i] + '.png', dpi=600)
                plt.show()

            if key == 'inter':
                inter_improve = g_acc_g - g_acc_s
            acc_single_threshold.append(g_acc_s)
            f1_single_threshold.append(g_f1_s)
            reca_single_threshold.append(g_reca_s)
            prec_single_threshold.append(g_prec_s)

            acc_group_threshold.append(g_acc_g)
            f1_group_threshold.append(g_f1_g)
            reca_group_threshold.append(g_reca_g)
            prec_group_threshold.append(g_prec_g)

            group_size.append(len(group_dict[key]))
        acc_group_threshold = np.array(acc_group_threshold)
        acc_single_threshold = np.array(acc_single_threshold)
        # group_f1 = np.array(group_f1)
        group_size = np.array(group_size)
        group_based_ave_acc = np.inner(acc_group_threshold, group_size) / group_size.sum()
        single_based_ave_acc = np.inner(acc_single_threshold, group_size) / group_size.sum()
        print(sim_list_str[i] + " single-based-ave-acc " + str(
            round(single_based_ave_acc * 100, 2)) + " group-based-ave-acc " + str(round(group_based_ave_acc * 100, 2)))
        ave_improvement_inter += inter_improve
        ave_improvement+=(group_based_ave_acc-single_based_ave_acc)
    print('average inter-class improvement is ', round(ave_improvement_inter / 8 * 100, 2))
    print('average overall improvement is ', round(ave_improvement/8*100, 2))
    return True

def process():
    # to keep the same testing set for using different ratio of training data,
    # we use 10% of data to evaluate the performance.
    test_path = partial_graph_path + \
                "%s_train_ratio_%s_test.json" % (dataset, "0.1")
    test_data = open(test_path).readlines()  # read test data only
    label_list = []
    group_dict = {}
    row_id = 0
    target_posterior_list = []
    reference_posterior_list = []
    feature_list = []
    for row in test_data:
        row = json.loads(row)
        label_list.append(row["label"]) # if there is an edge
        nl0, nl1 = row["gcn_pred0_label"], row["gcn_pred1_label"]
        if nl1 < nl0:
            nl0, nl1 = nl1, nl0
        group_key = str(nl0)+str(nl1)
        if group_key not in group_dict.keys():
            group_dict[group_key]=[]
        group_dict[group_key].append(row_id)
        row_id+=1

        target_posterior_list.append([row["gcn_pred0"], row["gcn_pred1"]])
        reference_posterior_list.append(
            [row["dense_pred0"], row["dense_pred1"]])
        feature_list.append([row["feature_arr0"], row["feature_arr1"]])

    sim_list_target = attack_0(target_posterior_list)
    # sim_list_target = attack_0_entropy(target_posterior_list)

    for i in range(len(sim_list_target)):
        sim_list_target[i]=(np.array(sim_list_target[i])/max(sim_list_target[i])).tolist()

    node_label_num = len(target_posterior_list[0][0])

            # print('~~~~~~~~~~acc: link stealing~~~~~~~~~~~~~')
              # write_acc(sim_list_target, label_list)
             # print('~~~~~~~~~~acc: group-based method~~~~~~~~~~~~~')
             # write_acc_group(sim_list_target, label_list, copy.deepcopy(group_dict), node_label_num, desc="target posterior similarity (group)")

    # print('~~~~~~~~~~auc: link stealing~~~~~~~~~~~~~')
    # write_auc(sim_list_target, label_list, desc="target posterior similarity")
    # print('~~~~~~~~~~auc: group-based method~~~~~~~~~~~~~')
    write_auc_group(sim_list_target, label_list, copy.deepcopy(group_dict), node_label_num, desc="target posterior similarity (group)")

    # viz(sim_list_target, label_list, copy.deepcopy(group_dict), node_label_num)

    # assert False

if __name__ == "__main__":
    # using_broad_threshold = True
    # using_broad_threshold = False
    process()
