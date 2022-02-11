import os
import sys
import numpy as np

from scipy.stats import mode
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn import datasets
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from graphviz import Source

# iris = datasets.load_iris()
# X = iris["data"][:, 2:]
# y = iris["target"]


# tree_clf = DecisionTreeClassifier(random_state=42, max_depth=2)
# tree_clf.fit(X, y)


# # 可视化决策树
# image_path = os.path.join("image")
# if not os.path.isdir(image_path):
#     os.makedirs(image_path)
#
# export_graphviz(
#     tree_clf,
#     out_file=os.path.join(image_path, "iris_tree.dot"),
#     feature_names=iris.feature_names[2:],
#     class_names=iris.target_names,
#     rounded=True,
#     filled=True
# )
# Source.from_file(os.path.join(image_path, "iris_tree.dot"))

# 绘制模型的决策边界
# def show_decision_boundary(clf, X, y, axes=None, is_iris=True, legend=False, show_training=True):
#     if axes is None:
#         axes = [0, 7.5, 0, 3]
#     x1s = np.linspace(axes[0], axes[1], 100)
#     x2s = np.linspace(axes[2], axes[3], 100)
#     x1, x2 = np.meshgrid(x1s, x2s)
#     X_new = np.c_[x1.ravel(), x2.ravel()]
#     y_pre = clf.predict(X_new).reshape(x1.shape)
#     custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
#     plt.contourf(x1, x2, y_pre, alpha=0.3, cmap=custom_cmap)
#     if not is_iris:
#         custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
#         plt.contour(x1, x2, y_pre, cmap=custom_cmap2, alpha=0.8)
#     if show_training:
#         plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris setosa")
#         plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris versicolor")
#         plt.plot(X[:, 0][y == 2], X[:, 1][y == 2], "g^", label="Iris virginica")
#         plt.axis(axes)
#
#     if is_iris:
#         plt.xlabel("Petal length", fontsize=16)
#         plt.ylabel("Petal width", fontsize=16)
#     else:
#         plt.xlabel(r"$x_1$", fontsize=16)
#         plt.ylabel(r"$x_2$", fontsize=16, rotation=0)
#
#     if legend:
#         plt.legend(loc="lower right", fontsize=16)


# plt.figure(figsize=(8, 4))
# show_decision_boundary(tree_clf, X, y)
# plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
# plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
# plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
# plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
# plt.text(1.40, 1.0, "Depth=0", fontsize=15)
# plt.text(3.2, 1.80, "Depth=1", fontsize=13)
# plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
# plt.show()

# 决策树模型的不稳定性
# 修改模型随机数之后生成新模型
# tree_clf = DecisionTreeClassifier(random_state=40, max_depth=2)
# tree_clf.fit(X, y)
# show_decision_boundary(tree_clf, X, y)
# plt.plot([0, 7.5], [0.8, 0.8], "k-", linewidth=2)
# plt.plot([0, 7.5], [1.75, 1.75], "k--", linewidth=2)
# plt.text(1, 0.9, "depth_1", fontsize=14)
# plt.text(1, 1.8, "depth_2", fontsize=14)
# plt.show()

# 绘制经过旋转数据集造成的模型不同
# np.random.seed(6)
# Xs = np.random.rand(100, 2) - 0.5
# ys = (Xs[:, 0] > 0).astype(np.float32) * 2
#
# # 旋转数据集
# angle = np.pi / 4
# rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
# Xsr = Xs.dot(rotation_matrix)
#
# tree_clf_1 = DecisionTreeClassifier(random_state=42)
# tree_clf_2 = DecisionTreeClassifier(random_state=42)
# tree_clf_1.fit(Xs, ys)
# tree_clf_2.fit(Xsr, ys)
#
# fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# plt.sca(axes[0])
# show_decision_boundary(tree_clf_1, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], is_iris=False)
# plt.sca(axes[1])
# show_decision_boundary(tree_clf_2, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], is_iris=False)
# plt.ylabel("")
# plt.show()


# 使用卫星数据集绘制使用超参数正则化的模型
# Xm, ym = datasets.make_moons(n_samples=100, noise=0.25, random_state=53)
#
# deep_clf_1 = DecisionTreeClassifier(random_state=42)
# deep_clf_2 = DecisionTreeClassifier(random_state=42, min_samples_leaf=4)
# deep_clf_1.fit(Xm, ym)
# deep_clf_2.fit(Xm, ym)
#
# fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# plt.sca(axes[0])
# show_decision_boundary(deep_clf_1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], is_iris=False)
# plt.title("No restriction", fontsize=16)
# plt.sca(axes[1])
# show_decision_boundary(deep_clf_2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], is_iris=False)
# plt.title("min_samples_leaf = 4", fontsize=16)
# plt.ylabel("")
# plt.show()

# 决策树回归
# def show_regression_predictions(tree_reg, X, y, axes=None, ylabel="$y$"):
#     if not axes:
#         axes = [0, 1, -0.2, 1]
#     x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
#     y_pre = tree_reg.predict(x1)
#
#     plt.axis(axes)
#     plt.xlabel("$x_1$", fontsize=14)
#     if ylabel:
#         plt.ylabel(ylabel, fontsize=14)
#
#     plt.plot(X, y, "b.")
#     plt.plot(x1, y_pre, "r.-", linewidth=2, label="$\hat{y}$")


# np.random.seed(42)
# m = 100
# X = np.random.rand(m, 1)
# y = 4 * (X - 0.5) ** 2
# y = y + np.random.randn(m, 1) / 10

# tree_reg_1 = DecisionTreeRegressor(random_state=42, max_depth=2)
# tree_reg_2 = DecisionTreeRegressor(random_state=42, max_depth=3)
# tree_reg_1.fit(X, y)
# tree_reg_2.fit(X, y)
#
# fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# plt.sca(axes[0])
# show_regression_predictions(tree_reg_1, X, y)
# # 分割线
# for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
#     plt.plot([split, split], [-0.2, 1], style, linewidth=2)
# plt.text(0.21, 0.65, "Depth=0", fontsize=14)
# plt.text(0.01, 0.2, "Depth=1", fontsize=14)
# plt.text(0.65, 0.8, "Depth=1", fontsize=14)
# plt.legend(loc="upper center", fontsize=16)
# plt.title("max_depth=2", fontsize=18)
#
# plt.sca(axes[1])
# show_regression_predictions(tree_reg_2, X, y)
# for split, style, line_width in (
#         (0.1973, "k-", 2), (0.0917, "k--", 2), (0.7718, "k--", 2), (0.0458, "k:", 1), (0.1298, "k:", 1),
#         (0.2873, "k:", 1), (0.9040, "k:", 1)):
#     plt.plot([split, split], [-0.2, 1], style, linewidth=line_width)
# plt.text(0.3, 0.5, "Depth=2", fontsize=13)
# plt.title("max_depth=3", fontsize=14)
# plt.show()


# 超参数正则化的模型对比
# tree_reg_1 = DecisionTreeRegressor(random_state=42)
# tree_reg_2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
#
# tree_reg_1.fit(X, y)
# tree_reg_2.fit(X, y)
#
# x1 = np.linspace(0, 1, 100).reshape(-1, 1)
# y_pre_1 = tree_reg_1.predict(x1)
# y_pre_2 = tree_reg_2.predict(x1)
#
# # 绘制图
# fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# plt.sca(axes[0])
# plt.plot(X, y, "b.")
# plt.plot(x1, y_pre_1, "r.-", linewidth=2, label="$\hat{y}$")
# plt.axis([0, 1, -0.2, 1.1])
# plt.xlabel("$x_1$", fontsize=14)
# plt.ylabel("$y$")
# plt.title("No restrictions", fontsize=18)
# plt.legend(loc="upper center")
#
# plt.sca(axes[1])
# plt.plot(X, y, "b.")
# plt.plot(x1, y_pre_2, "r.-", linewidth=2, label="$\hat{y}$")
# plt.axis([0, 1, -0.2, 1.1])
# plt.xlabel("$x_1$", fontsize=14)
# plt.title("min_samples_leaf=10")
# plt.show()


# 课后练习
# 7.为卫星数据集训练并微调一个决策树。
# a.使用make_moons（n_samples=10000，noise=0.4）生成一个卫
# 星数据集。

# b.使用train_test_split（）拆分训练集和测试集。

# c.使用交叉验证的网格搜索（在GridSearchCV的帮助下）为
# DecisionTreeClassifier找到适合的超参数。提示：尝试
# max_leaf_nodes的多种值。

# d.使用超参数对整个训练集进行训练，并测量模型在测试集上的
# 性能。你应该得到约85%～87%的准确率。
X, y = datasets.make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {"max_leaf_nodes": list(range(2, 100)), "min_samples_split": [2, 3, 4]}

grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
# print(grid_search_cv.best_estimator_)
# y_test_pre = grid_search_cv.predict(X_test)
# score = accuracy_score(y_test, y_test_pre)
# print(score)

# 8.按照以下步骤种植森林。
# a.继续之前的练习，生产1000个训练集子集，每个子集包含随机
# 挑选的100个实例。提示：使用Scikit-Learn的ShuffleSplit来实现。

# b.使用前面得到的最佳超参数值，在每个子集上训练一个决策
# 树。在测试集上评估这1000个决策树。因为训练集更小，所以这些决
# 策树的表现可能比第一个决策树要差一些，只能达到约80%的准确率。

# c.见证奇迹的时刻到了。对于每个测试集实例，生成1000个决策
# 树的预测，然后仅保留次数最频繁的预测（可以使用SciPy的mode（）
# 函数）。这样你在测试集上可获得大多数投票的预测结果。

# d.评估测试集上的这些预测，你得到的准确率应该比第一个模型
# 更高（高出0.5%～1.5%）。恭喜，你已经训练出了一个随机森林分类
# 器！
n_trees = 1000
n_instances = 100

mini_sets = []

# 生成一千个数据集
rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

# 结果集
accuracy_score_list = []
# 复制一千个模型 并且预测取准确率的平均值
forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]
for tree_clf, (X_mini_train, y_mini_train) in zip(forest, mini_sets):

    tree_clf.fit(X_mini_train, y_mini_train)
    y_pre = tree_clf.predict(X_test)
    accuracy_score_list.append(accuracy_score(y_test, y_pre))


# 对测试集做了一个预测，取其中出现于最高的预测作为结果
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)


# 计算最后的准确率
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
score = accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
print(score)