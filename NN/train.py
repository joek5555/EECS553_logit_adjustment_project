# code for training the model

import torch
import numpy as np
import os
from model import FullConnectNN
import random
from plot import Plotter
import checkpoint as checkpoint
from dataset import MSCAD

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def predictions(logits):
    return torch.argmax(logits, dim=1)


def accuracy(y_true, y_pred):
    num_right = torch.count_nonzero((y_true == y_pred).int()).item()
    num_total = y_true.size(dim=0)
    return num_right/num_total


def _train_epoch(train_loader, model, criterion, optimizer, base_probabilities, config):
    for i, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X)
        if 'use_logits_adjustment' not in config: # use standard loss
            loss = criterion(output, y)
        else: # use logit adjustment as described in paper
            tau = 1.0
            output = output + torch.log((base_probabilities**tau + 1e-12).to(torch.float32)).repeat(output.shape[0], 1)
            loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, epoch, base_probabilities, config):
    """
    Evaluates the model on the train and validation set.
    """
    stat = []
    for data_loader in [val_loader, train_loader]:
        y_true, y_pred, running_loss = evaluate_loop(data_loader, model, base_probabilities, config, criterion)
        total_loss = np.sum(running_loss) / y_true.size(0)
        total_acc = accuracy(y_true, y_pred)
        stat += [total_acc, total_loss]
    plotter.stats.append(stat)
    plotter.log_nn_training(epoch)
    plotter.update_nn_training_plot(epoch)


def evaluate_loop(data_loader, model, base_probabilities, config, criterion=None):
    model.eval()
    y_true, y_pred, running_loss = [], [], []
    for X, y in data_loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            if criterion is not None:
                
                if 'use_logits_adjustment' not in config: # use standard loss
                    running_loss.append(criterion(output, y).item() * X.size(0))
                else: # use logitd adjustment as described in paper
                    tau = 1.0
                    output = output + torch.log((base_probabilities**tau + 1e-12).to(torch.float32)).repeat(output.shape[0], 1)
                    running_loss.append(criterion(output, y).item() * X.size(0))
    model.train()
    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)
    return y_true, y_pred, running_loss


def train(config, model, train_loader, val_loader, base_probabilities):

    if 'use_weighted_loss' not in config:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        # weight the loss funtion using the reciprocal of the base probabilities
        # this ensures that labels that occur less frequently are weighted more
        criterion = torch.nn.CrossEntropyLoss(weight = (1/base_probabilities).to(torch.float32))


    learning_rate = config['learning_rate']
    momentum = config['momentum']
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Attempts to restore the latest checkpoint if exists
    print('Loading model...')
    force = config['ckpt_force'] if 'ckpt_force' in config else False
    model, start_epoch, stats = checkpoint.restore_checkpoint(model, config['ckpt_path'], force=force)

    # Create plotter
    plot_name = config['plot_name'] if 'plot_name' in config else 'NN'
    plotter = Plotter(stats, plot_name)

    # Evaluate the model
    _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, start_epoch, base_probabilities, config)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config['num_epoch']):
        # Train model on training set
        _train_epoch(train_loader, model, criterion, optimizer, base_probabilities, config)

        # Evaluate model on training and validation set
        _evaluate_epoch(plotter, train_loader, val_loader, model, criterion, epoch + 1, base_probabilities, config)

        # Save model parameters
        checkpoint.save_checkpoint(model, epoch + 1, config['ckpt_path'], plotter.stats)

    print('Finished Training')

    # Save figure and keep plot open
    plotter.save_nn_training_plot()
    plotter.hold_training_plot()



if __name__ == '__main__':

    config = {
        'data_path': 'data',
        'batch_size': 100,
        'num_epoch': 100,                 # number of epochs for training
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
    train_dataset = MSCAD(batch_size=config["batch_size"], dataset_path=os.path.join(config["data_path"], "train"))
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

    # necessary for logit adjustment
    base_probabilities = train_dataset.get_base_probabilities()

    # pass in the number of features to the FullConnectNN
    model = FullConnectNN(train_dataset.get_data_shape()[1])
    train(config, model, train_dataloader, val_dataloader, base_probabilities)
