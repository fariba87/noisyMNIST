# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:37:15 2022

@author: scc
"""

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes,average='micro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes,average='micro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes,average='micro')
print('F1 score: %f' % f1)
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)

 keras.metrics.Precision()
 
 
 # evaluate against test set
labels, predictions = [], []
for Xtest, Ytest in test_dataset:
Ytest_ = model.predict_on_batch(Xtest)
ytest = np.argmax(Ytest, axis=1)
ytest_ = np.argmax(Ytest_, axis=1)
labels.extend(ytest.tolist())
predictions.extend(ytest.tolist())
print("test accuracy: {:.3f}".format(accuracy_score(labels,
predictions)))
print("confusion matrix")
print(confusion_matrix(labels, predictions
                       
                       
                       # predict probabilities for test set
yhat_probs = model.predict(x_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(x_test, verbose=0)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes,average='micro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes,average='micro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes,average='micro')
print('F1 score: %f' % f1)
# kappa
kappa = cohen_kappa_score(y_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)


# evaluate the model
_, train_acc = model.evaluate(ax_train, ay_train, verbose=0)
_, test_acc = model.evaluate(ax_test, by_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))