import streamlit as st
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

# Function to generate dataset and calculate ROC AUC
def generate_dataset_and_roc_auc(weight):
    # generate 2 class dataset
    X, y = make_classification(n_samples=1000, n_classes=2, weights=[weight, 1 - weight], random_state=1)
    # split into train/test sets with same class ratio
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
    # no skill model, stratified random class predictions
    model = DummyClassifier(strategy='stratified')
    model.fit(trainX, trainy)
    yhat = model.predict_proba(testX)
    naive_probs = yhat[:, 1]
    # calculate roc auc
    roc_auc_naive = roc_auc_score(testy, naive_probs)
    # skilled model
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)
    yhat = model.predict_proba(testX)
    model_probs = yhat[:, 1]
    # calculate roc auc
    roc_auc_model = roc_auc_score(testy, model_probs)
    # plot roc curves
    plot_roc_curve(testy, naive_probs, model_probs)
    # print ROC AUC values
    print('No Skill ROC AUC: %.3f' % roc_auc_naive)
    print('Logistic ROC AUC: %.3f' % roc_auc_model)
    print('Weight', weight)

# plot no skill and model roc curves
def plot_roc_curve(test_y, naive_probs, model_probs):
    # plot naive skill roc curve
    fpr, tpr, _ = roc_curve(test_y, naive_probs)
    plt.plot(fpr, tpr, linestyle='--', label='No Skill')
    # plot model roc curve
    fpr, tpr, _ = roc_curve(test_y, model_probs)
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

# Create slider widget for the weights parameter
weight_slider = st.sidebar.slider(min=0, max=0.99, step=0.1, value=0.5, description='Weight:')
# Define function to update the plot when the slider value changes
def update_plot(change):
    weight = change.new
    generate_dataset_and_roc_auc(weight)
# Register the update_plot function to be called when the slider value changes
weight_slider.observe(update_plot, 'value')
