from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    true_positive_rate,
    selection_rate,
    count
)
import pandas as pd
from fairlearn.metrics import MetricFrame


def display_fairness(true_label, pred_label, sensitive_feat):
    metrics = {'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score,
               'false positive rate': false_positive_rate,
               'true positive rate': true_positive_rate,
               'selection rate': selection_rate,
               'count': count}
    metric_frame = MetricFrame(metrics=metrics,
                               y_true=pd.Series(true_label),
                               y_pred=pd.Series(pred_label),
                               sensitive_features=sensitive_feat)
    print("sr.overall:\n{}".format(metric_frame.overall))
    print("sr.by_group:\n{}".format(metric_frame.by_group))
    print("sr.sensitive_levels:\n{}".format(metric_frame.sensitive_levels))
