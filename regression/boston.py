import numpy as np
from featureScaling import featureScale
from regression import Regression

from sklearn.datasets import load_boston

boston = load_boston()

X = boston['data']
y = boston['target']
feature_names = boston['feature_names']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

reg = Regression()
reg.gradientDescent(X_train, y_train, None, 0.05)
print('test data: ',y_test[10:15])
print('predicted data: ',reg.predict(X_test[10:15]))

reg.normalEquation(X_train, y_train)
print('test data: ',y_test[10:15])
print('predicted data: ',reg.predict(X_test[10:15]))
