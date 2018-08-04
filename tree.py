#!/usr/bin/env python3

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import free_func
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plt = free_func.plot_decision_regions(X_combined, y_combined, classifier = tree, test_idx = range(105, 150))
plt.xlabel('Długość płatka [cm]')
plt.ylabel('Szeropkość płatka [cm]')
plt.legend(loc = 'upper left')
y_pred = tree.predict(X_test)
print('Dokładność: %.2f' % accuracy_score(y_test, y_pred))
plt.show()