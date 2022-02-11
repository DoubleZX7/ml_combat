import sklearn
import numpy as np
import xgboost
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier


# 展现即使是弱分类器，只要数量够多也会得到一个很好的结果
# heads_prob = 0.51
# coin_tosses = (np.random.rand(10000, 10) < heads_prob).astype(np.int32)
# cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)
#
# plt.figure(figsize=(8, 3.5))
# plt.plot(cumulative_heads_ratio)
# plt.plot([1, 10000], [0.5, 0.5], "k--", label="50%")
# plt.plot([1, 10000], [0.51, 0.51], "k-", label="51%")
# plt.xlabel("Number of coin tosses")
# plt.ylabel("Heads ratio")
# plt.legend(loc="lower right")
# plt.axis([0, 10000, 0.42, 0.58])
# plt.show()

# 训练一个投票分类器，由三种不同种类的分类器组成
# 通过修改SVC中probability参数与VotingClassifier中voting参数可以把硬投票分类器转换为软投票分类器，修改之后一般会得到一个很好的分数
# X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y)


#
# rand_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# log_clf = LogisticRegression(solver="lbfgs", random_state=42)
# svc_clf = SVC(gamma="scale", probability=True, random_state=42)
#
# voting_clf = VotingClassifier(
#     estimators=[("rand_clf", rand_clf), ("log_clf", log_clf), ("svc_clf", svc_clf)],
#     voting="soft"
# )
# # voting_clf.fit(X_train, y_train)
#
# for clf in (rand_clf, log_clf, svc_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_test_predict = clf.predict(X_test)
#     score = accuracy_score(y_test, y_test_predict)
#     print(clf.__class__.__name__, score)

# Bagging与Posting
# 训练一个有500棵决策树的分类器，与一个决策树分类器比较
# bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True,
#                             random_state=42)
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))


# tree_clf = DecisionTreeClassifier(random_state=42)
# tree_clf.fit(X_train, y_train)


# y_pred = tree_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))


# 比较两个分类器的决策边界
def show_clf_decision_boundary(clf, X, y, axes=None, alpha=0.5, contour=True):
    if not axes:
        axes = [-1.5, 2.45, -1, 1.5]
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=12)
    plt.ylabel(r"$x_2$", fontsize=12)


# fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# plt.sca(axes[0])
# show_clf_decision_boundary(tree_clf, X, y)
# plt.title("Decision Tree", fontsize=14)
# plt.sca(axes[1])
# show_clf_decision_boundary(bag_clf, X, y)
# plt.title("Decision Trees with Bagging", fontsize=14)
# plt.ylabel("")
# plt.show()


#

# bag_clf = BaggingClassifier(
#     DecisionTreeClassifier(), max_samples=100, n_estimators=100, bootstrap=True, oob_score=True, random_state=42
# )
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))

# 随机森林
# rand_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
# rand_clf.fit(X_train, y_train)
# rand_y_pred = rand_clf.predict(X_test)
#
# bag_clf = BaggingClassifier(DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16), n_estimators=500,
#                             random_state=42)
# bag_clf.fit(X_train, y_train)
# bag_y_pred = bag_clf.predict(X_test)
#
# print(np.sum(rand_y_pred == bag_y_pred) / len(rand_y_pred))

# 绘画15棵决策树的决策边界
# X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#
# plt.figure(figsize=(6, 4))
#
# for i in range(15):
#     tree_clf = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
#     rand_index = np.random.randint(0, len(X_train), len(X_train))
#     tree_clf.fit(X_train[rand_index], y_train[rand_index])
#     show_clf_decision_boundary(tree_clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.02, contour=False)
#
# plt.show()

# 特征重要性
# 使用iris数据计算特征的重要性
# iris = load_iris()
# X = iris["data"]
# y = iris["target"]
# rand_clf = RandomForestClassifier(n_estimators=500, random_state=42)
# rand_clf.fit(X, y)
#
# for name, score in zip(iris["feature_names"], rand_clf.feature_importances_):
#     print(name, score)

# mnist = fetch_openml("mnist_784", version=1, as_frame=False)
# mnist.target = mnist.target.astype(np.uint8)
# rand_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# X = mnist["data"]
# y = mnist["target"]
# rand_clf.fit(X, y)
#
#
# def show_digit(data):
#     image = data.reshape(28, 28)
#     plt.imshow(image, cmap=mpl.cm.hot, interpolation="nearest")
#     plt.axis("off")
#
#
# show_digit(rand_clf.feature_importances_)
# bar = plt.colorbar(ticks=[rand_clf.feature_importances_.min(), rand_clf.feature_importances_.max()])
# bar.ax.set_yticklabels(['Not important', 'Very important'])
# plt.show()

# AdaBoost
# X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm="SAMME.R",
#                              random_state=42)
# ada_clf.fit(X_train, y_train)
# show_clf_decision_boundary(ada_clf, X, y)
# plt.show()

# 绘制不同学习率SVM的决策边界
# X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# m = len(X_train)
#
# fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# for subplot, learning_rate in ((0, 1), (1, 0.5)):
#     plt.sca(axes[subplot])
#     sample_weights = np.ones(m) / m
#     for i in range(5):
#         svm_clf = SVC(kernel="rbf", C=0.2, gamma=0.6, random_state=42)
#         svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
#         y_pred = svm_clf.predict(X_train)
#
#         r = sample_weights[y_pred != y_train].sum() / sample_weights.sum()
#         alpha = learning_rate * np.log((1 - r) / r)
#         sample_weights[y_pred != y_train] *= np.exp(alpha)
#         sample_weights /= sample_weights.sum()
#
#         show_clf_decision_boundary(svm_clf, X, y, alpha=0.2)
#         plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
#     if subplot == 0:
#         plt.text(-0.75, -0.95, "1", fontsize=14)
#         plt.text(-1.05, -0.95, "2", fontsize=14)
#         plt.text(1.0, -0.95, "3", fontsize=14)
#         plt.text(-1.45, -0.5, "4", fontsize=14)
#         plt.text(1.36, -0.95, "5", fontsize=14)
#     else:
#         plt.ylabel("")
#
# plt.show()

# 梯度提升
# np.random.seed(42)
# X = np.random.rand(100, 1) - 0.5
# y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)


# tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
# tree_reg.fit(X, y)
#
# y2 = y - tree_reg.predict(X)
#
# tree_reg_2 = DecisionTreeRegressor(max_depth=2, random_state=42)
# tree_reg_2.fit(X, y2)
#
# y3 = y2 - tree_reg_2.predict(X)
# tree_reg_3 = DecisionTreeRegressor(max_depth=2, random_state=42)
# tree_reg_3.fit(X, y3)
#
#
# # X_new = np.array([[0.8]])
# #
# # y_pred = sum(tree.predict(X_new) for tree in (tree_clf, tree_clf_2, tree_clf_3))
#
#
def show_predictions(regs, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(reg.predict(x1.reshape(-1, 1)) for reg in regs)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)


#
#
# plt.figure(figsize=(11, 11))
#
# plt.subplot(321)
# show_predictions([tree_reg], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-",
#                  data_label="Training set")
# plt.ylabel("$y$", fontsize=16, rotation=0)
# plt.title("Residuals and tree predictions", fontsize=16)
#
# plt.subplot(322)
# show_predictions([tree_reg], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
# plt.ylabel("$y$", fontsize=16, rotation=0)
# plt.title("Ensemble predictions", fontsize=16)
#
# plt.subplot(323)
# show_predictions([tree_reg_2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+",
#                  data_label="Residuals")
# plt.ylabel("$y - h_1(x_1)$", fontsize=16)
#
# plt.subplot(324)
# show_predictions([tree_reg, tree_reg_2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
# plt.ylabel("$y$", fontsize=16, rotation=0)
#
# plt.subplot(325)
# show_predictions([tree_reg_3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
# plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
# plt.xlabel("$x_1$", fontsize=16)
#
# plt.subplot(326)
# show_predictions([tree_reg, tree_reg_2, tree_reg_3], X, y, axes=[-0.5, 0.5, -0.1, 0.8],
#                  label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
# plt.xlabel("$x_1$", fontsize=16)
# plt.ylabel("$y$", fontsize=16, rotation=0)
#
# plt.show()

# 尝试使用GradientBoostingRegressor还原上述操作
# gb_reg_3 = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
# gb_reg_3.fit(X, y)
# X_new = np.array([[0.8]])
# print(gb_reg_3.predict(X_new))

# 比较决策树数量不同的决策边界
# gb_reg_500 = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
# gb_reg_500.fit(X, y)
#
# fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# plt.sca(axes[0])
# show_predictions([gb_reg_3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
# plt.title("learning_rate={}, n_estimators={}".format(gb_reg_3.learning_rate, gb_reg_3.n_estimators), fontsize=14)
# plt.xlabel("$x_1$", fontsize=16)
# plt.ylabel("$y$", fontsize=16, rotation=0)
#
# plt.sca(axes[1])
# show_predictions([gb_reg_500], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
# plt.title("learning_rate={}, n_estimators={}".format(gb_reg_500.learning_rate, gb_reg_500.n_estimators), fontsize=14)
# plt.xlabel("$x_1$", fontsize=16)
#
# plt.show()

# 训练120棵决策树，找到误差最低的那棵树
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49)
# gb_reg = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
# gb_reg.fit(X_train, y_train)
#
# errors = [mean_squared_error(y_test, y_pred) for y_pred in gb_reg.staged_predict(X_test)]
# bst_n_estimators = np.argmin(errors) + 1
#
# best_bg_reg = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
# best_bg_reg.fit(X_train, y_train)
#
#
# min_err = np.min(errors)
#
# fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
# plt.sca(axes[0])
# plt.plot(np.arange(1, len(errors) + 1), errors, "b.-")
# plt.plot([bst_n_estimators, bst_n_estimators], [0, min_err], "k--")
# plt.plot([0, bst_n_estimators], [min_err, min_err], "k--")
# plt.plot(bst_n_estimators, min_err, "ko")
# plt.text(bst_n_estimators, min_err*1.2, "Minimum", ha="center", fontsize=14)
# plt.axis([0, 120, 0, 0.01])
# plt.xlabel("Number of trees")
# plt.ylabel("Error", fontsize=16)
# plt.title("Validation error", fontsize=14)
#
# plt.subplot(122)
# show_predictions([best_bg_reg], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
# plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)
# plt.ylabel("$y$", fontsize=16, rotation=0)
# plt.xlabel("$x_1$", fontsize=16)
#
# plt.show()

# 提前停止训练
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=49)
# gb_reg = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)
#
# min_val_error = float("inf")
# error_going_up = 0
# for n_estimators in range(1, 120):
#     gb_reg.n_estimators = n_estimators
#     gb_reg.fit(X_train, y_train)
#     y_pred = gb_reg.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     if mse < min_val_error:
#         min_val_error = mse
#         error_going_up = 0
#     else:
#         error_going_up += 1
#         if error_going_up == 5:
#             break
#
# print(gb_reg.n_estimators)
# print(min_val_error)

# 使用using
# xgb_reg = xgboost.XGBRegressor(random_state=42)
# xgb_reg.fit(X_train, y_train)
# y_pred = xgb_reg.predict(X_test)
# print(mean_squared_error(y_test, y_pred))


# 课后练习
# 8.加载MNIST数据集（第3章中有介绍），将其分为一个训练集、
# 一个验证集和一个测试集（例如，使用50 000个实例训练、10 000个
# 实例验证、10 000个实例测试）。然后训练多个分类器，比如一个随
# 机森林分类器、一个极端随机树分类器和一个SVM分类器。接下来，尝
# 试使用软投票法或者硬投票法将它们组合成一个集成，这个集成在验
# 证集上的表现要胜过它们各自单独的表现。成功找到集成后，在测试
# 集上测试。与单个的分类器相比，它的性能要好多少？
# mnist = fetch_openml("mnist_784", version=1, as_frame=False)
# X = mnist.data
# y = mnist.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=42)
#
# rand_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# extra_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
# svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
# mlp_clf = MLPClassifier(random_state=42)
#
# estimators = [rand_clf, extra_clf, svm_clf, mlp_clf]
# for estimator in estimators:
#     estimator.fit(X_train, y_train)

# scores = [estimator.score(X_val, y_val) for estimator in estimators]

# estimator_list = [
#     ("rand_clf", rand_clf),
#     ("extra_clf", extra_clf),
#     ("svm_clf", svm_clf),
#     ("mlp_clf", mlp_clf)
# ]
# voting_clf = VotingClassifier(estimator_list)
# voting_clf.fit(X_train, y_train)
# print(voting_clf.score(X_val, y_val))

# 我们尝试删除集合里面的支持向量机分类器看一下会不会提高性能
# voting_clf.set_params(svm_clf=None)
# del voting_clf.estimators[2]
# print(voting_clf.estimators)
# voting_clf.fit(X_train, y_train)
# print(voting_clf.score(X_val, y_val))

# 尝试修改投票方式
# voting_clf.voting = "soft"
# print(voting_clf.score(X_val, y_val))

# 发现硬投票的性能更好一点
# 使用测试集来测试性能
# print(voting_clf.score(X_test, y_test))
# print([estimator[1].score(X_test, y_test) for estimator in voting_clf.estimators_])

# 9.运行练习题8中的单个分类器，用验证集进行预测，然后用预测
# 结果创建一个新的训练集：新训练集中的每个实例都是一个向量，这
# 个向量包含所有分类器对于一张图像的一组预测，目标值是图像的
# 类。恭喜，你成功训练了一个混合器，结合第一层的分类器，它们一
# 起构成了一个stacking集成。现在在测试集上评估这个集成。对于测
# 试集中的每张图像，使用所有的分类器进行预测，然后将预测结果提
# 供给混合器，得到集成的预测。与前面训练的投票分类器相比，这个
# 集成的结果如何？
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data
y = mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=42)

rand_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [rand_clf, extra_clf, svm_clf, mlp_clf]
for estimator in estimators:
    estimator.fit(X_train, y_train)

X_val_prediction = np.empty((len(X_val), len(estimators)))

for index, estimator in enumerate(estimators):
    X_val_prediction[:, index] = estimator.predict(X_val)

rand_clf_last = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rand_clf_last.fit(X_val_prediction, y_val)

print(rand_clf_last.oob_score_)

# 把模型放入测试集测试
X_test_prediction = np.empty((len(X_test), len(estimators)))
for index, estimator in enumerate(estimators):
    X_test_prediction[:, index] = estimator.predict(X_test)

y_last_pred = rand_clf_last.predict(X_test_prediction)
print(accuracy_score(y_test, y_last_pred))

# 最后得到的堆叠模型性能不如投票模型性能好，甚至低于个别单个模型的性能



