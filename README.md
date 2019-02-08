# accelerometerClassification

This is a toy machine learning problem with classifying body postures and movements of people wearing accelerometer devices from a publicly available dataset.

A description of the dataset and the csv file with class assignment can be foun here: https://archive.ics.uci.edu/ml/datasets/Wearable+Computing%3A+Classification+of+Body+Postures+and+Movements+%28PUC-Rio%29#

#### Description of What the Code Does

- 'accelerometerCalssification.py' imports the data from the csv file, than splits it into training and testing sets. The data goes  through several preprocessing steps (scaling, filling in NaNs with averages, etc), and Python's sklearn library is used to compare classification performance with a logistic regression model in addition to an svm model. Some error analysis plots are created to try to determine whether there is enough training data for the number and informativeness of the given features in addition to testing different regularizer parameter values for each of the two models. 

#### Error Analysis

##### Effects of Training Size on Performance

