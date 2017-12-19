#!/usr/bin/env python
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

print "\n\nLoading mnist data ..."
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"].astype('float32'), mnist["target"].astype('int')

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

print "\n\nGenerating training and testing datasets ..."
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print "\n\nScaling data ..."
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print "\n\nBuild tf dnn classifier ..."
# https://www.tensorflow.org/api_docs/python/tf/contrib/learn/infer_real_valued_columns_from_input
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

# https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], feature_columns = feature_columns,n_classes = 10)

print "\n\nTraining ..."
# dnn_clf.fit(input_fn=lambda: input_fn(X))
dnn_clf.fit(x=X_train, y=y_train, batch_size=64, steps=40000)


print "\n\nTesting ..."
y_pred = list(dnn_clf.predict(X_test))
print accuracy_score(y_test, y_pred)

print "\n\nEvaluating ..."
print dnn_clf.evaluate(X_test, y_test)
