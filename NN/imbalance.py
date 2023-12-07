# code for running advanced metrics on model

from dataset import MSCAD
from train import evaluate_loop, train
import torch
from model import FullConnectNN
import numpy as np
import os
import checkpoint as checkpoint

def per_class_accuracy(y_true, y_pred, num_classes):
    """
    Compute the per-class accuracy given true and predicted labels.
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
        - num_classes: the number of different classes, int
    Returns:
        - per_class_acc: per-class accuracy, list of floats
    """
    per_class_acc = []
    for class_num in range(num_classes):
        y_pred_mask = (y_pred == class_num).int()
        y_true_mask = (y_true == class_num).int()
        TP_and_TN = torch.count_nonzero((y_pred_mask == y_true_mask).int()).item()
        num_total = y_true.shape[0]
        per_class_acc.append(TP_and_TN/num_total)

    return per_class_acc


def precision(y_true, y_pred, num_classes):
    """
    Compute the per class precision where each class is considered the positive class for 
    one precision value 
    Precision = TP / (TP + FP)
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
        - num_classes: the number of different classes, int
    Returns:
        - prec: precision, list
    """
    per_class_precision = []
    for class_num in range(num_classes):
        y_pred_mask = (y_pred == class_num).int()
        y_true_mask = (y_true == class_num).int()
        TP =  torch.count_nonzero(torch.logical_and(y_true_mask, y_pred_mask).int()).item()
        FP = torch.count_nonzero(torch.logical_and(torch.logical_not(y_true_mask), y_pred_mask).int()).item()
        if TP == 0 and FP == 0:
            per_class_precision.append(0.0)
        else:
            per_class_precision.append(TP/(TP+FP))

    return per_class_precision



def recall(y_true, y_pred, num_classes):
    """
    Compute the per class recall, where each class is considered the positive class
    for one recall value
    Recall = TP / (TP + FN)
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
        - num_classes: the number of different classes, int 
    Returns:
        - rec: recall, list
    """

    per_class_recall = []
    for class_num in range(num_classes):
        y_pred_mask = (y_pred == class_num).int()
        y_true_mask = (y_true == class_num).int()
        TP =  torch.count_nonzero(torch.logical_and(y_true_mask, y_pred_mask).int()).item()
        FN = torch.count_nonzero(torch.logical_and(torch.logical_not(y_pred_mask), y_true_mask).int()).item()
        per_class_recall.append(TP/(TP+FN))

    return per_class_recall


def f1_score(y_true, y_pred, num_classes):
    """
    Compute the per class f1 score, where each class is considered the positive class
    for one f1 score
    F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
        - num_classes: the number of different classes, int
    Returns:
        - f1: f1-score, list
    """

    prec_array = np.array(precision(y_true, y_pred, num_classes))
    rec_array = np.array(recall(y_true, y_pred, num_classes))

    f1 = 2*(prec_array * rec_array) / (prec_array + rec_array)
    
    return f1.tolist()


def compute_metrics(val_dataloader, model, base_probabilities, config):
    y_true, y_pred, _ = evaluate_loop(val_dataloader, model, base_probabilities, config)
    print('Per-class accuracy: ', [round(value, 4) for value in per_class_accuracy(y_true, y_pred, num_classes=6)])
    print('Precision: ', [round(value, 4) for value in precision(y_true, y_pred, num_classes=6)])
    print('Recall: ', [round(value,4) for value in recall(y_true, y_pred, num_classes=6)])
    print('F1-score: ', [round(value,4) for value in f1_score(y_true, y_pred, num_classes=6)])


# run to get the metrics on a model saved by a checkpoint for a specific model
if __name__ == '__main__':
    
    config = {
        'data_path': 'data',
        'batch_size': 100,
        'num_epoch': 25,                 # number of epochs for training
        'learning_rate': 1e-3,           # learning rate
        'momentum': 0.9,                  # momentum 

        'plot_name': 'MSCAD_unweighted',
        'ckpt_path': 'checkpoints/unweighted',  # directory to save our model checkpoints


        # 'plot_name': 'MSCAD_weighted',
        # 'ckpt_path': 'checkpoints/weighted',  # directory to save our model checkpoints
        # 'use_weighted_loss': True,
        
        # 'plot_name': 'MSCAD_logit_adjust',
        # 'use_logits_adjustment': True,
        # 'ckpt_path': 'checkpoints/logit_adjust',  # directory to save our model checkpoints
    }

    # create the datasets for the training and validation split
    train_dataset = MSCAD(batch_size=100, dataset_path=os.path.join(config["data_path"], "train"))
    val_dataset = MSCAD(batch_size=["batch_size"], dataset_path=os.path.join(config["data_path"], "val"))

    # determine which features have 0 standard deviation in both the training and validation split
    # then remove these features from both datasets
    invalid_features = np.unique(np.concatenate((train_dataset.get_index_std_0(), val_dataset.get_index_std_0()), 0))
    train_dataset.normalize(invalid_features)
    val_dataset.normalize(invalid_features)

    # check to ensure that the training dataset and validation dataset has the same number of features
    if train_dataset.get_data_shape()[1] != val_dataset.get_data_shape()[1]:
        print("ERROR: Training and validation data does not have the same number of features")
        print(train_dataset.get_data_shape())
        print(val_dataset.get_data_shape())
        raise

    # create the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

    model = FullConnectNN(train_dataset.get_data_shape()[1])
    # necessary for logit adjustment
    base_probabilities = train_dataset.get_base_probabilities()

    model, start_epoch, stats = checkpoint.restore_checkpoint(model, config["ckpt_path"], force=False)

    compute_metrics(val_dataloader, model, base_probabilities, config)
