import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 生成文章中的第一个图大间隔的分类
iris = datasets.load_iris()
# X = iris["data"][:, (2, 3)]
# y = iris["target"]
#
# setosa_or_versicolor = (y == 1) | (y == 0)
# X = X[setosa_or_versicolor]
# y = y[setosa_or_versicolor]
#
# svm_clf = SVC(kernel="linear", C=float("inf"))
#
# svm_clf.fit(X, y)

# 拟合三个预测
# x0 = np.linspace(0, 5.5, 200)
# pred_1 = 5 * x0 - 20
# pred_2 = x0 - 1.8
# pred_3 = 0.1 * x0 + 0.5


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
    y = -w[0] / w[1] * x0 - b / w[1]
    # 斜率
    k = -w[0] / w[1]
    # 上边界
    # 因为方程y = kx + b 所以 b = y - kx
    support_vectors = sv[-1]
    b = support_vectors[1] - support_vectors[0] * k
    y_up = k * x0 + b
    # 下边界
    support_vectors = sv[0]
    b = support_vectors[1] - support_vectors[0] * k
    y_down = k * x0 + b
    # 支持向量
    plt.scatter(sv[:, 0], sv[:, 1], s=180, facecolors="#FFAAAA")
    # 分割线、上边界、下边界
    plt.plot(x0, y, "k-", linewidth=2)
    plt.plot(x0, y_up, "k--", linewidth=2)
    plt.plot(x0, y_down, "k--", linewidth=2)


# fig = plt.figure(figsize=(10, 4))
# # 绘制随便拟合的图
# fig.add_subplot(121)
# plt.plot(x0, pred_1, "g--", linewidth=2)
# plt.plot(x0, pred_2, "m-", linewidth=2)
# plt.plot(x0, pred_3, "r-", linewidth=2)
# # 画实例
# plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris versicolor")
# plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris setosa")
# plt.xlabel("Patel length", fontsize=14)
# plt.ylabel("Patel width", fontsize=14)
# plt.legend(loc="upper left", fontsize=14)
# plt.axis([0, 5.5, 0, 2])
#
# # 绘制模型的图
# fig.add_subplot(122)
# show_svc_decision_boundary(svm_clf, 0, 5.5)
# # 画实例
# plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
# plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo")
# plt.xlabel("Patel length", fontsize=14)
# plt.axis([0, 5.5, 0, 2])
# plt.show()

# 画图查看归一化对SVM的影响
# 创建一个特征数值差距很大的特征
# Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
# ys = np.array([0, 0, 1, 1])
# svm_clf = SVC(kernel="linear", C=100)
# svm_clf.fit(Xs, ys)
#
# # 绘画一个没有使用特征归一化的图
# fig = plt.figure(figsize=(10, 3))
# fig.add_subplot(121)
# show_svc_decision_boundary(svm_clf, 0, 6)
# # 绘画实例
# plt.plot(Xs[:, 0][ys == 1], Xs[:, 1][ys == 1], "bo")
# plt.plot(Xs[:, 0][ys == 0], Xs[:, 1][ys == 0], "ms")
# plt.xlabel("$x_0$", fontsize=20)
# plt.ylabel("$x_1$", fontsize=20, rotation=0)
# plt.title("Unscaled", fontsize=16)
# plt.axis([0, 6, 0, 90])
#
# # 绘画第二个使用了特征归一化的图
# fig.add_subplot(122)
# sca = StandardScaler()
# X_sca = sca.fit_transform(Xs)
# svm_clf.fit(X_sca, ys)
# plt.plot(X_sca[:, 0][ys == 1], X_sca[:, 1][ys == 1], "bo")
# plt.plot(X_sca[:, 0][ys == 0], X_sca[:, 1][ys == 0], "ms")
# show_svc_decision_boundary(svm_clf, -2, 2)
# plt.xlabel("$x'_0$", fontsize=20)
# plt.ylabel("$x'_1$  ", fontsize=20, rotation=0)
# plt.title("Scaled", fontsize=16)
# plt.axis([-2, 2, -2, 2])
# plt.show()

# 软件间隔分类
# 添加异常值
# X_outlier = np.array([[3.4, 1.3], [3.2, 0.8]])
# y_outlier = np.array([0, 0])
# Xo1 = np.concatenate([X, X_outlier[:1]], axis=0)
# yo1 = np.concatenate([y, y_outlier[:1]], axis=0)
# Xo2 = np.concatenate([X, X_outlier[1:]], axis=0)
# yo2 = np.concatenate([y, y_outlier[1:]], axis=0)
#
# # 绘制第一个显示异常值的图
# fig = plt.figure(figsize=(10, 3))
# fig.add_subplot(121)
# plt.plot(Xo1[:, 0][yo1 == 1], Xo1[:, 1][yo1 == 1], "bs")
# plt.plot(Xo1[:, 0][yo1 == 0], Xo1[:, 1][yo1 == 0], "yo")
# plt.text(0.3, 1.0, "Impossible!", fontsize=24, color="red")
# plt.xlabel("Petal length", fontsize=16)
# plt.ylabel("Petal width", fontsize=16)
# plt.annotate("Outlier",
#              xy=(X_outlier[0][0], X_outlier[0][1]),
#              xytext=(2.5, 1.7),
#              ha="center",
#              arrowprops=dict(facecolor='black', shrink=0.1),
#              fontsize=16
#              )
# plt.axis([0, 5.5, 0, 2])
#
# # 绘制第二个没有使用软间隔的模型
# fig.add_subplot(122)
# svm_clf = SVC(kernel="linear", C=10 ** 9)
# svm_clf.fit(Xo2, yo2)
# show_svc_decision_boundary(svm_clf, 0, 5.5)
# plt.plot(Xo2[:, 0][yo2 == 1], Xo2[:, 1][yo2 == 1], "bs")
# plt.plot(Xo2[:, 0][yo2 == 0], Xo2[:, 1][yo2 == 0], "yo")
# plt.annotate("Outlier", xy=(X_outlier[1][0], X_outlier[1][1]), xytext=(3.2, 0.08),
#              ha="center",
#              arrowprops=dict(facecolor='black', shrink=0.1),
#              fontsize=16)
# plt.axis([0, 5.5, 0, 2])
# plt.show()


# 第五章的第一段实例代码
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.int32)

# svm_clf = Pipeline([
#     ("sca", StandardScaler()),
#     ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42))
# ])
# svm_clf.fit(X, y)
# pred = svm_clf.predict([[5.5, 1.7]])
# print(pred)

# 比较超参数C值不同的SVM模型
# 训练两个模型超参数C值不同
# sca = StandardScaler()
# svm_clf_1 = LinearSVC(C=1, loss="hinge", random_state=42)
# svm_clf_2 = LinearSVC(C=100, loss="hinge", random_state=42)
# svm_clf_1 = SVC(kernel="linear", C=1)
# svm_clf_2 = SVC(kernel="linear", C=float("inf"))
# svm_clf_2.fit(X, y)
# sca_svm_clf_1 = Pipeline([
#     ("sca", sca),
#     ("linear_svc", svm_clf_1)
# ])
#
# sca_svm_clf_2 = Pipeline([
#     ("sca", sca),
#     ("linear_svc", svm_clf_2)
# ])
#
# sca_svm_clf_1.fit(X, y)
# sca_svm_clf_2.fit(X, y)
#
# # 由于LinearSVC没有支持向量，所以需要自己计算支持向量
# b1 = svm_clf_1.decision_function([-sca.mean_ / sca.scale_])
# b2 = svm_clf_2.decision_function([-sca.mean_ / sca.scale_])
# w1 = svm_clf_1.coef_[0] / sca.scale_
# w2 = svm_clf_2.coef_[0] / sca.scale_
# svm_clf_1.intercept_ = np.array([b1])
# svm_clf_2.intercept_ = np.array([b2])
# svm_clf_1.coef_ = np.array([w1])
# svm_clf_2.coef_ = np.array([w2])
#
# t = y * 2 - 1
# support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
# support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
# svm_clf_1.support_vectors_ = X[support_vectors_idx1]
# svm_clf_2.support_vectors_ = X[support_vectors_idx2]
#
# # 绘画两个不通超参数的图
# fig = plt.figure(figsize=(13, 3))
# fig.add_subplot(121)
# show_svc_decision_boundary(svm_clf_1, 4, 6)
# plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="Iris virginica")
# plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs", label="Iris versicolor")
# plt.legend(loc="upper left", fontsize=14)
# plt.xlabel("Patel length", fontsize=14)
# plt.ylabel("Patel width", fontsize=14)
# plt.grid(True)
# plt.title(f"C= {svm_clf_1.C}")
# plt.axis([4, 6, 0.8, 2.8])
#
#
# fig.add_subplot(122)
# show_svc_decision_boundary(svm_clf_2, 4, 6)
# plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
# plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
# plt.xlabel("Patel length", fontsize=14)
# plt.grid(True)
# plt.title(f"C= {svm_clf_2.C}")
# plt.axis([4, 6, 0.8, 2.8])
# plt.show()


# 非线性支持向量机
X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
X2D = np.c_[X1D, X1D**2]
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

# 绘画第一个线性不可分的数据
fig = plt.figure(figsize=(10, 3))
fig.add_subplot(121)
plt.grid(True, which="both")
plt.axhline(y=0, color="k")
plt.plot(X1D[:, 0][y == 0], np.zeros(4), "bs")
plt.plot(X1D[:, 0][y == 1], np.zeros(5), "g^")
plt.gca().get_yaxis().set_ticks([])
plt.xlabel(r"$x_1$", fontsize=20)
plt.axis([-4.5, 4.5, -0.2, 0.2])

# 绘制第二个线性可分的数据图
fig.add_subplot(122)
plt.grid(True, which="both")
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.plot(X2D[:, 0][y == 0], X2D[:, 1][y == 0], "bs")
plt.plot(X2D[:, 0][y == 1], X2D[:, 1][y == 1], "y^")
plt.xlabel(r"$x_1$", fontsize=14)
plt.ylabel(r"$x_2$  ", fontsize=14, rotation=0)
plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
plt.axis([-4.5, 4.5, -1, 17])
plt.show()

