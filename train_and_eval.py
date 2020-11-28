# Python Script to train and evaluate different machine learning models to:
#
#   -   Perform classification and predict the class of the 'answered_correctly'.
#       The testing performance was evaluated using True accuracy (TA), F1 score (F1) and AUC
#
# Usage:        python train_and_eval.py PATH_TO_DATA_CSV
#           ie. python train_and_eval.py data
#
# Output:       Prints out testing performance of the model/s
#               Also, info relating to the state of the script is printed to STDOUT for the user to track if need be
#
# Strat so far:
#
#   - Only 2 features are being used so far:
#           * prior_question_elapsed_time (dataframe col_index = 8)
#           * prior_question_had_explanation (dataframe col_index = 9)
#
#   - Other important features that should be considerd
#           * timestamp: I'm pretty sure that the time at which the student answered the question is important.
#               since taking a while between questions may suggest that the student is struggling or
#               is attempting to answer a hard question
#           * we could rank questions by difficulty, by assessing how all prior students so far are doing on the question
#               and this rank could be used as a training feature. 
#           * question_tag: which is a tag to cluster questions together, a student might do well on certain tags,
#               as opposed to other tags
#           * we could measure how many questions a particular student has got right so far 

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Set the plots to be (7 inches x 7 inches)
rcParams['figure.figsize'] = 7,7

# Classification models
class_models = {
    'logr' : LogisticRegression(random_state=10, warm_start=True),
    'dt': DecisionTreeClassifier(max_depth=5),
    'rf': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, warm_start=True)
}
'''
    'knc': KNeighborsClassifier(3),
    'svc': SVC(kernel="linear", C=0.025),
    'svc_g': SVC(gamma=2, C=1),
    'gr': GaussianProcessClassifier(1.0 * RBF(1.0)),
    'mlp': MLPClassifier(alpha=1, max_iter=1000),
    'ada': AdaBoostClassifier(),
    'gnb': GaussianNB(),
    'qda': QuadraticDiscriminantAnalysis()
'''

# Classification output feature
class_output = 'answered_correctly'

# Feature selection process that uses the correlation between the input and output via the chosen 
# features. The higher correlating features are chosen when considering a subset of these features. 
def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    
    # Calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    
    # Obtain feature names
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    return cor_feature, cor_list

# function to plot auc per epoch for each classifier 
def plot_auc_per_classifier(epochs, results_auc):
    for key, value in results_auc.items():
        plt.plot(epochs, value, label=key)
    plt.title('Testing AUC socres classifier^-1 epoch^-1')
    plt.xlabel('Epoch', fontsize=1)
    plt.ylabel('AUC', fontsize=8)
    plt.legend()

    plt.show()
    plt.clf()

# function to plot performance metrics across all different model for different number of
# 'significant' input features used for training the algorithms
def plot_all_metric_all_feat_combos(mae, metric, all_feats, models, output):
    plt.title('Testing ' + metric + ' for different models and features, output= '+ output)
    plt.xlabel('Model', fontsize=8)
    plt.ylabel(metric, fontsize=8)

    for li in mae:
        if metric == 'TA' or metric == 'F1':
            plt.scatter(list(models.keys()), li)
        else:
            plt.plot(list(models.keys()), li)
    
    plt.legend([(str(i)+' sign feat.') for i in range(1, len(all_feats)+1)])
    plt.show()

# function to train the different models and plot their testing performance
def train_and_eval(X, y, ep, sub_ep, results_auc):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    
    # cycle through different classifiers
    for key, value in class_models.items():
        # train model
        model = value.fit(X_train, y_train)

        if sub_ep == 102:
            # evaluate model on the last batch of the epoch only
            preds = model.predict(X_test)

            ta = accuracy_score(y_test, preds)
            print(f'Epoch: {ep}, sub-epoch: {sub_ep}, {key} test accuracy = {ta}')

            f1 = f1_score(y_test, preds, average='weighted')
            print(f'Epoch: {ep}, sub-epoch: {sub_ep}, {key} test F1 = {f1}')

            # get raw prob scores for auc 
            probs = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, probs[:,1])
            if key in results_auc:
                results_auc[key].append(auc)
            else:
                results_auc[key] = [auc]
            print(f'Epoch: {ep}, sub-epoch: {sub_ep}, {key} test auc = {auc}')
    return results_auc

def preprocess_df(df_data):
    # remove rows with NaN (missing) values
    df_data_temp = df_data.copy()
    df_data_temp = df_data_temp[df_data_temp.answered_correctly != -1]
    
    train_feats = []
    # Initially, I only consider the last 2 features from the 'train.csv' dataset so far, change at will
    train_feats_indices = [8,9]

    for feat_i in train_feats_indices:
        train_feats.append(df_data_temp.columns[feat_i])
        df_data_temp[df_data_temp.columns[feat_i]] = df_data_temp[df_data_temp.columns[feat_i]].fillna(0)       

    X = df_data_temp[train_feats]
    y = df_data_temp[class_output]

    return X, y

# Main function
def main():
    # Load the data
    print('Loading data...')

    results_auc = {}
    epochs = range(1,11)
    sub_epochs = range(1, 103)

    for ep in epochs:
        for sub_ep in sub_epochs:
            df_data = pd.read_csv('data/train_part_'+str(sub_ep)+'.csv')

            # Preprocess data
            #print('Preprocessing data...')
            X, y = preprocess_df(df_data)

            # Train and evaluate models for regression task
            #print('Training and Evaluating models...')
            results_auc = train_and_eval(X, y, ep, sub_ep, results_auc)
            print(f'Epoch: {ep}, sub-epoch: {sub_ep}')

    plot_auc_per_classifier(epochs, results_auc)

# Start of program
main()