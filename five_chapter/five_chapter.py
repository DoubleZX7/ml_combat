import numpy as np

from sklearn import datasets
from sklearn.svm import SVC

# 生成文章中的第一个图大间隔的分类
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = iris["target"]

setosa_or_versicolor = (y == 1) | (y == 0)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

svm_clf = SVC(kernel="linear", C=float("inf"))

svm_clf.fit(X, y)

# 拟合三个预测
x0 = np.linspace(0, 5.5, 200)
pred_1 = 5 * x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5


def show_svc_decision_boundary(model, x_min, x_max):
    """
    绘制支持向量
    :param model:模型
    :param x_min:最小的X轴值
    :param x_max:最大的X轴值
    :return:
    """
    # 系数
    w = model.coef_[0]
    # 截距
    b = model.intercept_[0]
    # 支持向量
    sv = model.support_vectors_
    x0 = np.linspace(x_min, x_max)
    # 因为 w0 * x0 + w1 * y + b = 0 所以 -w1y = w0 * x0 + b  所以y = w0 * x0 / -w1 + b / -w1
    # 分割线
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]
    # 斜率
    k = -w[0] / w[1]
    # 上边界
    # 因为方程y = kx + b 所以 b = y - kx
    support_vectors = sv[1]
    b = support_vectors[1] - support_vectors[0] * k
    y_up = k * x0 + b
    # 下边界
    support_vectors = sv[0]
    b = support_vectors[1] - support_vectors[0] * k
    y_down = k * x0 + b
    print(y_up)
    print(y_down)


show_svc_decision_boundary(svm_clf, 0, 5.5)
