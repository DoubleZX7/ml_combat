import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC, LinearSVC, LinearSVR, SVR
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import reciprocal, uniform

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
# X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
# X2D = np.c_[X1D, X1D**2]
# y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
#
# # 绘画第一个线性不可分的数据
# fig = plt.figure(figsize=(10, 3))
# fig.add_subplot(121)
# plt.grid(True, which="both")
# plt.axhline(y=0, color="k")
# plt.plot(X1D[:, 0][y == 0], np.zeros(4), "bs")
# plt.plot(X1D[:, 0][y == 1], np.zeros(5), "g^")
# plt.gca().get_yaxis().set_ticks([])
# plt.xlabel(r"$x_1$", fontsize=20)
# plt.axis([-4.5, 4.5, -0.2, 0.2])
#
# # 绘制第二个线性可分的数据图
# fig.add_subplot(122)
# plt.grid(True, which="both")
# plt.axhline(y=0, color="k")
# plt.axvline(x=0, color="k")
# plt.plot(X2D[:, 0][y == 0], X2D[:, 1][y == 0], "bs")
# plt.plot(X2D[:, 0][y == 1], X2D[:, 1][y == 1], "y^")
# plt.xlabel(r"$x_1$", fontsize=14)
# plt.ylabel(r"$x_2$  ", fontsize=14, rotation=0)
# plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
# plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
# plt.axis([-4.5, 4.5, -1, 17])
# plt.show()

#  分类非线性的卫星数据集
X, y = datasets.make_moons(n_samples=100, noise=0.15, random_state=42)


def show_datasets(X, y, axis):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "y^")
    plt.axis(axis)
    plt.grid(True, which="both")
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)


# show_datasets(X, y, [-1.5, 2.5, -1, 1.5])
# plt.show()

# 文中的第二段示例代码用管道制作一个包括多项式归一化的SVM模型
# polynomial_svm_clf = Pipeline([
#     ("pl", PolynomialFeatures(degree=3)),
#     ("sac", StandardScaler()),
#     ("svm_clf", LinearSVC(random_state=42, C=10, loss="hinge"))
# ])
# polynomial_svm_clf.fit(X, y)


# 绘制非线性模型的预测图
def show_pred(clf, axis):
    x0s = np.linspace(axis[0], axis[1], 100)
    x1s = np.linspace(axis[1], axis[2], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


# show_pred(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
# show_datasets(X, y, [-1.5, 2.5, -1, 1.5])
# plt.show()

# 比较超参数不同的模型
# 训练两个不同超参数的SVM模型
# poly_svm_clf = Pipeline([
#     ("sca", StandardScaler()),
#     ("svm", SVC(kernel="poly", C=5, degree=3, coef0=1))
# ])
# poly_svm_clf.fit(X, y)
#
# poly_100_svm_clf = Pipeline([
#     ("sca", StandardScaler()),
#     ("svm", SVC(kernel="poly", C=5, degree=10, coef0=100))
# ])
# poly_100_svm_clf.fit(X, y)
#
# # 画两个超参数不同的模型
# fig = plt.figure(figsize=(10, 3))
# fig.add_subplot(121)
# show_datasets(X, y, [-1.5, 2.4, -1, 1.5])
# show_pred(poly_svm_clf, [-1.5, 2.4, -1, 1.5])
# plt.title("d = 3, r = 1, C = 5", fontsize=18)
#
# fig.add_subplot(122)
# show_datasets(X, y, [-1.5, 2.4, -1, 1.5])
# show_pred(poly_100_svm_clf, [-1.5, 2.45, -1, 1.5])
# plt.title("d = 10, r = 100, C = 5", fontsize=18)
# plt.ylabel("")
# plt.show()


# 解决非线性的另外一个方法，添加相似特征
def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1) ** 2)


# gamma = 0.3
# X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
#
# x1s = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)
# x2s = gaussian_rbf(x1s, -2, gamma)
# x3s = gaussian_rbf(x1s, 1, gamma)
#
# XK = np.c_[gaussian_rbf(X1D, -2, gamma), gaussian_rbf(X1D, 1, gamma)]
# yk = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
#
# # 绘制两个图
# fig = plt.figure(figsize=(10, 3))
# fig.add_subplot(121)
# plt.grid(True, which="both")
# plt.axhline(y=0, color="k")
# plt.scatter([-2, 1], [0, 0], s=150, alpha=0.5, c="red")
# plt.plot(X1D[:, 0][yk == 1], np.zeros(5), "y^")
# plt.plot(X1D[:, 0][yk == 0], np.zeros(4), "bs")
# # 绘制两条相似特征
# plt.plot(x1s, x2s, "g--")
# plt.plot(x1s, x3s, "b:")
# plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
# plt.xlabel(r"$x_1$", fontsize=14)
# plt.ylabel(r"Similarity", fontsize=14)
# # 随机找一个X点
# plt.annotate(r'$\mathbf{x}$',
#              xy=(X1D[3, 0], 0),
#              xytext=[-0.5, 0.2],
#              ha="center",
#              arrowprops=dict(facecolor='black', shrink=0.1),
#              fontsize="14"
#              )
# plt.text(-2, 0.9, "$x_2$", ha="center", fontsize=18)
# plt.text(1, 0.9, "$x_3$", ha="center", fontsize=18)
# plt.axis([-4.5, 4.5, -0.1, 1.1])
#
#
# fig.add_subplot(122)
# plt.axhline(y=0, color="k")
# plt.axvline(x=0, color="k")
# plt.grid(True, which="both")
# plt.plot(XK[:, 0][yk == 0], XK[:, 1][yk == 0], "bs")
# plt.plot(XK[:, 0][yk == 1], XK[:, 1][yk == 1], "y^")
# plt.xlabel("$x_2$", fontsize=14)
# plt.ylabel("$x_3$", fontsize=14)
# plt.annotate(r'$\phi\left(\mathbf{x}\right)$',
#              xy=(XK[3, 0], XK[3, 1]),
#              xytext=(0.65, 0.50),
#              ha="center",
#              arrowprops=dict(facecolor='black', shrink=0.1),
#              fontsize="14"
#              )
# plt.plot([-0.1, 1.1], [0.57, -0.1], "r--", linewidth=3)
# plt.axis([-0.1, 1.1, -0.1, 1.1])
# plt.show()


# 使用高斯径向基函数核做SVC的内核训练模型
# svm_rbf_clf = Pipeline([
#     ("sca", StandardScaler()),
#     ("svm_clf", SVC(C=0.001, gamma=5, kernel="rbf"))
# ])
# svm_rbf_clf.fit(X, y)


# 绘制不同超参数的不同模型
# gamma1, gamma2 = 0.1, 5
# C1, C2 = 0.001, 1000
#
# hyper_params = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)
#
# svm_clfs = []
# for gamma, C in hyper_params:
#     svm_rbf_clf = Pipeline([
#         ("sca", StandardScaler()),
#         ("svm_clf", SVC(C=C, gamma=gamma, kernel="rbf"))
#     ])
#     svm_rbf_clf.fit(X, y)
#     svm_clfs.append(svm_rbf_clf)
#
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), sharex=True, sharey=True)
#
# for i, clf in enumerate(svm_clfs):
#     plt.sca(axes[i // 2, i % 2])
#     show_datasets(X, y, [-1.5, 2.45, -1, 1.5])
#     show_pred(clf, [-1.5, 2.45, -1, 1.5])
#     gamma, C = hyper_params[i]
#     plt.title(f"gamma = {gamma}, C = {C}", fontsize=16)
#     if i in (0, 1):
#         plt.xlabel("")
#     if i in (1, 3):
#         plt.ylabel("")
#
# plt.show()


# SVM回归
# np.random.seed(42)
# m = 50
# X = 2 * np.random.rand(m, 1)
# y = (4 + X ** 3 + np.random.rand(m, 1)).ravel()
#
# svm_reg = LinearSVR(epsilon=1.5, random_state=42)
# svm_reg.fit(X, y)
#
# # 训练两个不同超参数的SVM回归模型
# svm_reg_1 = LinearSVR(epsilon=1.5, random_state=42)
# svm_reg_2 = LinearSVR(epsilon=0.5, random_state=42)
# svm_reg_1.fit(X, y)
# svm_reg_2.fit(X, y)


def find_support_vectors(svm_reg, X, y):
    """
    寻找模型的支持向量
    :param svm_reg:
    :param X:
    :param y:
    :return:
    """
    y_pred = svm_reg.predict(X)
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
    return np.argwhere(off_margin)


# svm_reg_1.support_ = find_support_vectors(svm_reg_1, X, y)
# svm_reg_2.support_ = find_support_vectors(svm_reg_2, X, y)
#
# eps_x1 = 1
# eps_y_pred = svm_reg_1.predict([[eps_x1]])


# print(svm_reg_1.predict())


def show_svm_reg(reg, X, y, axis):
    xls = np.linspace(axis[0], axis[1], 100).reshape(100, 1)
    y_pre = reg.predict(xls)
    plt.plot(xls, y_pre, "k", linewidth=2, label=r"$\hat{y}$")
    plt.plot(xls, y_pre + reg.epsilon, "k--")
    plt.plot(xls, y_pre - reg.epsilon, "k--")
    plt.scatter(X[reg.support_], y[reg.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.axis(axis)


# fig, axes = plt.subplots(ncols=2, figsize=(10, 3), sharey=True)
# plt.sca(axes[0])
# show_svm_reg(svm_reg_1, X, y, [0, 2, 3, 11])
# plt.title(f"$\epsilon = {svm_reg_1.epsilon}$", fontsize=16)
# plt.ylabel(r"$y$", fontsize=18, rotation=0)
# plt.annotate(
#     '', xy=(eps_x1, eps_y_pred), xycoords='data',
#     xytext=(eps_x1, eps_y_pred - svm_reg_1.epsilon),
#     textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
# )
# plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
#
# plt.sca(axes[1])
# show_svm_reg(svm_reg_2, X, y, [0, 2, 3, 11])
# plt.title(f"$\epsilon = {svm_reg_2.epsilon}$", fontsize=16)
# plt.show()

# 非线性SVM回归
# np.random.seed(42)
# m = 100
# X = 2 * np.random.rand(m, 1) - 1
# y = (0.2 + 0.1 * X + 0.5 * X**2 + np.random.randn(m, 1)/10).ravel()
#
# # 训练两个不同超参数的模型
# svm_reg_1 = SVR(kernel="poly", C=100, degree=2, epsilon=0.1, gamma="scale")
# svm_reg_2 = SVR(kernel="poly", C=0.01, degree=2, epsilon=0.1, gamma="scale")
# svm_reg_1.fit(X, y)
# svm_reg_2.fit(X, y)
#
# # 绘制两个SVM回归模型
# fig, axes = plt.subplots(ncols=2, figsize=(10, 3), sharey=True)
# plt.sca(axes[0])
# show_svm_reg(svm_reg_1, X, y, [-1, 1, 0, 1])
# plt.title(f"$degree=2, C=100, \epsilon = 0.1$")
# plt.ylabel("$y_1$", fontsize=16)
#
# plt.sca(axes[1])
# show_svm_reg(svm_reg_2, X, y, [-1, 1, 0, 1])
# plt.title(f"$degree=2, C=0.01, \epsilon = 0.1$")
# plt.show()


# 课后练习
# 8.在一个线性可分离数据集上训练LinearSVC。然后在同一数据集
# 上训练SVC和SGDClassifier。看看你是否可以用它们产生大致相同的
# 模型。
# 寻找一个线性可分数据
# iris = datasets.load_iris()
# X = iris["data"][:, (2, 3)]
# y = iris["target"]
#
# index = (y == 1) | (y == 0)
# X = X[index]
# y = y[index]
#
# # 训练三个模型
# C = 5
# alpha = 1 / (C * len(X))
# line_clf = LinearSVC(C=C, loss="hinge", random_state=42)
# sgd_clf = SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001, alpha=alpha, max_iter=1000, tol=1e-3,
#                         random_state=42)
# svc_clf = SVC(kernel="linear", C=C)
#
# # 归一化
# sca = StandardScaler()
# X_sca = sca.fit_transform(X)
# # 拟合训练数据
# line_clf.fit(X_sca, y)
# sgd_clf.fit(X_sca, y)
# svc_clf.fit(X_sca, y)
#
# # 计算斜率（k）与偏差（b）
# k1 = -line_clf.coef_[0, 0] / line_clf.coef_[0, 1]
# b1 = -line_clf.intercept_ / line_clf.coef_[0, 1]
# k2 = -sgd_clf.coef_[0, 0] / sgd_clf.coef_[0, 1]
# b2 = -sgd_clf.intercept_ / sgd_clf.coef_[0, 1]
# k3 = -svc_clf.coef_[0, 0] / svc_clf.coef_[0, 1]
# b3 = -svc_clf.intercept_ / svc_clf.coef_[0, 1]
#
# # 将归一化之后的数据转换为原始数据
# line_1 = sca.inverse_transform([[-10, -10 * k1 + b1], [10, 10 * k1 + b1]])
# line_2 = sca.inverse_transform([[-10, -10 * k2 + b2], [10, 10 * k2 + b2]])
# line_3 = sca.inverse_transform([[-10, -10 * k3 + b3], [10, 10 * k3 + b3]])
#
# # 画图
# plt.figure(figsize=(10, 4))
# plt.plot(line_1[:, 0], line_1[:, 1], "k:", label="LinearSVC")
# plt.plot(line_2[:, 0], line_2[:, 1], "b--", label="SGD")
# plt.plot(line_3[:, 0], line_3[:, 1], "r-", label="SVC")
# plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "y^")
# plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
# plt.xlabel("Petal length", fontsize=16)
# plt.ylabel("Petal width", fontsize=16)
# plt.legend(loc="upper center", fontsize=16)
# plt.axis([0, 5.5, 0, 2])
# plt.show()

# 9.在MNIST数据集上训练SVM分类器。由于SVM分类器是个二元分类
# 器，所以你需要使用一对多来为10个数字进行分类。你可能还需要使
# 用小型验证集来调整超参数以加快进度。最后看看达到的准确率是多
# 少？
# data_path = ""
# mnist = loadmat("./data/mnist-original.mat")
# X, y = mnist["data"], mnist["label"]
#
# # 对数据进行转换，打乱顺序
# all_data = np.vstack((X, y))
# all_data = all_data.T
# np.random.shuffle(all_data)
# X = all_data[:, range(784)]
# y = all_data[:, 784]
# y = y.astype(np.uint8)
#
# X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]

# 最原始的SVM得分
# svc_clf = LinearSVC(random_state=42)
# svc_clf.fit(X_train, y_train)
# y_pre = svc_clf.predict(X_train)
# svc_score = accuracy_score(y_train, y_pre)
# print(svc_score)

# 加入归一化
# sca = StandardScaler()
# X_sca_train = sca.fit_transform(X_train)
# X_sca_test = sca.fit_transform(X_test)
# line_clf = LinearSVC(random_state=42)
# line_clf.fit(X_sca_train, y_train)
# y_pred = line_clf.predict(X_sca_train)
# line_scour = accuracy_score(y_train, y_pred)
# print(line_scour)

# 使用SVC的内核
# svc_clf = SVC()
# svc_clf.fit(X_sca_train[:10000], y[:10000])
# y_pred = svc_clf.predict(X_sca_train)
# svc_source = accuracy_score(y_train, y_pred)
# print(svc_source)

# 使用随机搜索寻找SVC的最优超参数
# svc_clf = SVC()
# param = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
# rand_search_cv = RandomizedSearchCV(svc_clf, param_distributions=param, verbose=2, cv=3)
# rand_search_cv.fit(X_sca_train[:1000], y_train[:1000])
#
#
# best_model = rand_search_cv.best_estimator_
# best_model.fit(X_sca_train, y_train)
# y_pred = best_model.predict(X_sca_train)
# best_source = accuracy_score(y_train, y_pred)
# print(rand_search_cv.best_estimator_)
# print(best_source)
#
# y_test_pre = best_model.predict(X_sca_test)
# best_test_source = accuracy_score(y_test, y_test_pre)
# print(best_test_source)

# 10.在加州住房数据集上训练一个SVM回归模型。
housing = datasets.fetch_california_housing()
X = housing["data"]
y = housing["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 归一化
sca = StandardScaler()
X_sca_train = sca.fit_transform(X_train)

# 训练LinearSVR模型
# line_reg = LinearSVR(random_state=42)
# line_reg.fit(X_sca_train, y_train)
# y_pre = line_reg.predict(X_sca_train)
# line_reg_mse = mean_squared_error(y_train, y_pre)
# print(line_reg_mse)

# 最基础的SVR
svr = SVR()
svr.fit(X_sca_train, y_train)
y_pre = svr.predict(X_sca_train)
mse = mean_squared_error(y_train, y_pre)
print(np.sqrt(mse))

y_test_pre = svr.predict(X_test)
mse = mean_squared_error(y_test, y_test_pre)
print(np.sqrt(mse))

# 训练SVR类模型并使用随机搜索寻找最优超参数
param = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rand_svm_reg = RandomizedSearchCV(SVR(), param_distributions=param, cv=3, n_iter=10, verbose=2)
rand_svm_reg.fit(X_sca_train, y_train)
best_model = rand_svm_reg.best_estimator_
y_pre = best_model.predict(X_train)
svm_reg_source = mean_squared_error(y_train, y_pre)
print(np.sqrt(svm_reg_source))

y_test_pre = best_model.predict(X_test)
svm_reg_test_source = mean_squared_error(y_test, y_test_pre)
print(np.sqrt(svm_reg_test_source))
