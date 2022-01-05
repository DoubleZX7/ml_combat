from sklearn import datasets
from sklearn.svm import SVC

# 生成文章中的第一个图大间隔的分类
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = iris["target"]
# print(len(X))
# print(len(y))
# print(X[0])

setosa_or_versicolor = (y == 1) | (y == 0)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)
