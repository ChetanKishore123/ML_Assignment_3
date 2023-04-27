import streamlit as st
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)

st.title("Hierarchical Clustering")

st.write("Create a slider for each step in hierarchical clustering of unsupervised data. Use the sklearn.datasets make_blobs function to create the dataset and visualize the predictions using a changing voronoi diagram.")

weight = st.sidebar.slider("imbalance ratio", min_value=0.1, max_value=0.9)

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

roc_auc_model = roc_auc_score(testy, model_probs)

fpr, tpr, _ = roc_curve(testy, naive_probs)

plt.plot(fpr, tpr, linestyle='--', label='No Skill')
# plot model roc curve
fpr, tpr, _ = roc_curve(testy, model_probs)
plt.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# plt.figure()
# ax = plt.Subplot(fig, 111)
fig = plt.plot()

# plt.box(False)
# plt.show()
st.pyplot(fig)
