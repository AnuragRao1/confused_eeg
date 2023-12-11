import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
from matplotlib import pyplot as plt

eeg = pd.read_csv('EEG_data.csv')
    
def get_group_labels(inputs, y_slice, groups):
    # group is defined as a combination between subject and video, groups is the group of every timepoint. y_slice is the true labels 
    # return output probabilities per group
    pred_prob = np.zeros(len(pd.unique(groups)))
    labels = np.zeros(len(pd.unique(groups)))
    for i in range(len(pd.unique(groups))):
        group = pd.unique(groups)[i]
        indices = np.where(groups == group)
        pred_prob[i] = np.mean(inputs[indices[0]])
        labels[i] = pd.unique(y_slice[indices[0]])[0]
    return pred_prob, labels

def prob_to_binary(arr, threshold=0.5):
    # convert probability array into labels based on a given threshold
    bin = np.copy(arr)
    for i in range(len(bin)):
        if arr[i] >= threshold:
            bin[i] = 1
        else:
            bin[i] = 0
    return bin

def plot_roc_curve(train_preds, train_labels, test_preds, test_labels, model_name):
    train_auc = roc_auc_score(train_labels, train_preds)
    test_auc = roc_auc_score(test_labels, test_preds)

    tr_fpr, tr_tpr,_ = roc_curve(train_labels, train_preds)
    te_fpr, te_tpr,thresholds = roc_curve(test_labels, test_preds)


    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.plot(tr_fpr, tr_tpr, label = f"Train (AUC = {train_auc})")
    plt.plot(te_fpr, te_tpr, label = f"Test (AUC = {test_auc})")
    plt.legend()
    plt.show()
    plt.clf()

def get_thresholds(preds, labels):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    tnr = 1 - fpr
    print("True negative rate: ", tnr)
    print("True positive rate: ", tpr)
    print("Thresholds: ", thresholds)
    return tnr, tpr, thresholds


