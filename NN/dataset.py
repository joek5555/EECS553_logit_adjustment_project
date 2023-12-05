# class to store dataset in format so that torch dataloader can be used
# call dataset_name = MSCAD(batch_size, dataset_path)
# then call dataset_name.normalize_data(features_to_remove)
# idea is that the features with 0 standard deviation are the same value for every single row
# therefore they can be removed
# so create your train_dataset and val_dataset, then use 
# invalid_features = np.unique(train_dataset.get_index_std_0(), val_dataset.get_index_std_0())
# to get the indices of the features that should be removed from the training and validation dataset
# then do
# train_dataset.normalize(invalid_features)
# val_dataset.normalize(invalid_features)


import os
import numpy as np
import torch


class MSCAD:
    
    def __init__(self, batch_size=4, dataset_path='data/train'):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.data_np = np.load(os.path.join(self.dataset_path, "features.npy"))
        self.labels_np = np.load(os.path.join(self.dataset_path, "labels.npy"))
        self.x_mean, self.x_std = self.compute_train_statistics()
        

    # required for torch dataloader
    def __getitem__(self,idx):
        return self.data[idx, :], self.labels[idx]
    
    # required for torch dataloader
    def __len__(self): 
        return self.data.shape[0]

        

    def compute_train_statistics(self):
        x_mean = np.mean(self.data_np, axis=0)
        x_std = np.std(self.data_np, axis=0)
        return x_mean, x_std
    
    def get_data_shape(self):
        return self.data_np.shape
    
    # function that returns the index of where the standard deviation of features is 0
    # if the std of a feature is 0, this means that the feature value is the same in 
    # every single row and the feature can be ignored
    def get_index_std_0(self):
        return np.where(self.x_std == 0)[0]
    
    # function to return how likely each label is to occur in the dataset
    # required for logit adjustment
    def get_base_probabilities(self):
        unique, counts = np.unique(self.labels_np, return_counts=True)
        base_probabilities = counts / counts.sum()
        base_probabilities = torch.from_numpy(base_probabilities)
        return base_probabilities
    
    def normalize(self, features_to_remove = None):
        # if no features are specified to remove, then just remove the features that have 0 standard deviation 
        if features_to_remove is None:
            features_to_remove = np.where(self.x_std == 0)[0]
        self.data_np = np.delete(self.data_np, features_to_remove, axis=1)
        self.x_mean, self.x_std = self.compute_train_statistics()
        self.data_normalized_np = (self.data_np - self.x_mean)/self.x_std
        self.data = torch.from_numpy(self.data_normalized_np).to(torch.float32)
        self.labels = torch.from_numpy(self.labels_np)
