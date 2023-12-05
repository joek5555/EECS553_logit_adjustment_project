# EECS553_logit_adjustment_project
Final project for EECS 553 Machine Learning at the University of Michigan. Logit adjustment for imbalanced tabular data

First, download the [MSCAD.csv file](https://www.kaggle.com/datasets/drjamailalsawwa/mscad).

Then run through the data_preprocessing.ipynb to generate the train and validation data splits.

Then run train.py to train a model, using the config dictionary to specify parameters.

Finally, use imbalance.py to calculate per class accuracy, precision, recall, and f1-score on the model parameters for a specific epoch.

## Future Improvements

Could use the Adam optimizer.

Can change the Fully connected NN shape or structure. 
