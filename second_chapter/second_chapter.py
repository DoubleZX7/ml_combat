import hashlib
import os
import urllib.request
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from scipy.stats import randint, expon, reciprocal
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV, \
    RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# pd.set_option("display.max_columns", None)

# 获取数据
data_path = os.path.join("data", "housing")
# if not os.path.exists(data_path):
#     os.makedirs(data_path)
# data_url = "https://github.com/ageron/handson-ml2/blob/master/datasets/housing/housing.csv"
data_path = os.path.join(data_path, "housing.csv")
# urllib.request.urlretrieve(data_url, data_path)
# 这里是数据暂时无法获取可以用我的地址下载数据

# 加载数据
housing_data = pd.read_csv(data_path)

# 探索数据
# print(housing_data.shape)
# print(housing_data.head())
# print(housing_data.info())
# print(housing_data.describe())
# 查看其中的object类型数据
# print(housing_data["ocean_proximity"].value_counts())


# 绘制直方图
# housing_data.hist(bins=50, figsize=(20, 15))
# plt.show()

# 创建测试集
# 为了每次的分数的测试集是一样的 需要先定一个固定的随机数

# np.random.seed(42)


def split_train_test(data, test_ratio: float):
    """
    按比例
    :param data: 所有数据
    :param test_ratio:测试数据的占比
    :return:
    """
    new_data = np.random.permutation(len(data))
    test_count = int(len(data) * test_ratio)
    test_index = new_data[:test_count]
    train_index = new_data[test_count:]
    return data.iloc[train_index], data.iloc[test_index]


train_data, test_data = split_train_test(housing_data, 0.2)
print(len(train_data))
print(len(test_data))


def check_id(identifier, test_ratio: float, h=hashlib.md5):
    """
    根绝has值大小分类
    :param identifier:选择标识符
    :param test_ratio: 测试集的占比
    :param h: 计算hash值的方式
    :return:
    """
    return bytearray(h(np.int64(identifier)).digest())[-1] < test_ratio * 256


def split_train_test_by_identifier(data, test_ratio, clo_name: str):
    """
    根据标识符的hash值来获取测试集
    :param data:原始数据
    :param test_ratio:测试集占比
    :param clo_name:列的名称
    :return:
    """
    d = data[clo_name].apply(lambda _id: check_id(_id, test_ratio))
    return data[~d], data[d]


# 使用index作为唯一标识符分割数据
# data_index = housing_data.reset_index()
# train_data, test_data = split_train_test_by_identifier(data_index, 0.2, "index")
# print(len(train_data))
# print(len(test_data))

# 使用经纬度作为唯一标识分割测试集和训练集
# housing_data["id"] = housing_data["longitude"] * 1000 + housing_data["latitude"]
# train_data, test_data = split_train_test_by_identifier(housing_data, 0.2, "id")
# print(len(train_data))
# print(len(test_data))

# 使用sklearn自带的方法分割训练集与测试集
# train_data, test_data = train_test_split(housing_data, test_size=0.2, random_state=42)
# print(len(train_data))
# print(len(test_data))


# 对收入中位数特征进行优化
print(housing_data.info())
housing_data["median_income"].hist()
plt.show()
housing_data["income_cut"] = pd.cut(housing_data["median_income"], bins=[0., 1.5, 3, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
print(housing_data["income_cut"].value_counts())
housing_data["income_cut"].hist()
plt.show()

# 根绝收入类别来进行分层抽样
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_data, housing_data["income_cut"]):
    start_train_set = housing_data.loc[train_index]
    start_test_set = housing_data.loc[test_index]

# 检查分层的比例
print(housing_data["income_cut"].value_counts() / len(housing_data))
train_data, test_data = train_test_split(housing_data, test_size=0.2, random_state=42)

# 计算纯随机抽样与分层抽样的抽样偏差
# def check_proportion(data):
#     return data["income_cut"].value_counts() / len(data)
#
#
# proportion = pd.DataFrame({
#     "All": check_proportion(housing_data),
#     "Random": check_proportion(train_data),
#     "Hierarchy": check_proportion(start_train_set)
# })

# proportion["Random_Err"] = (proportion["Random"] - proportion["All"])
# proportion["Hierarchy_Err"] = (proportion["Hierarchy"] - proportion["All"])
# print(proportion)

# 删除用于分层的收入分类
for set_ in (start_train_set, start_test_set):
    set_.drop("income_cut", axis=1, inplace=True)

# 把数据可视化
housing = start_train_set.copy()
housing_test = start_test_set.copy()
# 绘制散点图
# housing.plot(kind="scatter", x="longitude", y="latitude")
# plt.show()

# 查看到高密度地区分布
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

# 结合房价中位数来绘制散点图
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"] / 100, label="population",
             figsize=(10, 7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
# plt.legend()
# plt.show()


# 寻找特征相关性
print(housing.info())
corr = housing.corr()
print(corr["median_house_value"].sort_values(ascending=False))
# 选择相关性前三绘制散布矩阵
choose = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[choose], figsize=(12, 8))
plt.show()

# 着重看一下收入中位数与房价中位数的关系
# housing.plot(kind="scatter", y="median_house_value", x="median_income", alpha=0.1)
# plt.axis([0, 16, 0, 550000])
# plt.show()

# 尝试属性的组合
# 平均一户有多少个房间、平均一户有多少个卧室，平均一户有多少个人, 卧室在总房间数的占比
housing["households_per_room"] = housing["total_rooms"] / housing["households"]
housing["households_per_bedrooms"] = housing["total_bedrooms"] / housing["households"]
housing["households_per_population"] = housing["population"] / housing["households"]
housing["room_per_bedrooms"] = housing["total_bedrooms"] / housing["total_rooms"]

# 再次查看相关性
corr = housing.corr()
print(corr["median_house_value"].sort_values(ascending=False))

# 着重查看一个房子里的房间数特征与房价中位数的相关性
print(housing["households_per_room"].sort_values())
housing.plot(kind="scatter", x="households_per_room", y="median_house_value", alpha=0.1)
plt.axis([0, 5, 0, 520000])
plt.show()


# 为机器学习算法做数据准备
# 删除训练数据中的房间中位数特征  生成标签
housing_label = housing["median_house_value"].copy()
housing.drop("median_house_value", axis=1, inplace=True)

# 数据清洗
# 对于缺失的总的卧室总数有三种方法可以解决
# 1、删除卧室总数为空的区域
# housing.dropna(subset=["total_bedrooms"], inplace=True)
# 2、删除卧室总数这个属性
# housing.drop(["total_bedrooms"], axis=1, inplace=True)
# 3、使用中位数填补数据
# housing["total_bedrooms"].fillna(housing["total_bedrooms"].median, inplace=True)

# 取出数据中有缺失值的数据逐一演示每一个方法
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print(sample_incomplete_rows)
print(sample_incomplete_rows.dropna(subset=["total_bedrooms"]))
print(sample_incomplete_rows.drop("total_bedrooms", axis=1))
print(sample_incomplete_rows.fillna(sample_incomplete_rows["total_bedrooms"].median))

# 使用sklearn自带的方法处理缺失值
print(housing.info())
housing_num = housing.drop("ocean_proximity", axis=1)
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(housing_num)
print(X)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
print(housing.loc[sample_incomplete_rows.index.values])
print(housing.head())


# 处理文本和分类属性
# 取出数据中的文本数据
print(housing.info())
ocean_proximity = housing[["ocean_proximity"]]
print(ocean_proximity.head())

# 使用sklearn自带的OrdinalEncoder将字符串便成为数字
oe = OrdinalEncoder()
ocean_proximity_num = oe.fit_transform(ocean_proximity)
print(ocean_proximity_num[:10])

# 使用sklearn自带的OneHotEncoder将字符串生成独热向量
ohe = OneHotEncoder(sparse=False)
ocean_proximity_num = ohe.fit_transform(ocean_proximity)
print(ocean_proximity_num.toarray())

# 直接生成数组 放弃稀疏举证
ocean_proximity_num = ohe.fit_transform(ocean_proximity)
print(ocean_proximity_num)

# 自定义转换器
# print(housing.info())
# rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
# 动态获取字段下标
col_name = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [housing.columns.get_loc(i) for i in col_name]


# 用于做特征转换的转换器
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedroom_per_room: bool = True):
        self.add_bedroom_per_room = add_bedroom_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 平均一户有多少个房间、平均一户有多少个卧室，平均一户有多少个人, 卧室在总房间数的占比
        room_per_households = X[:, rooms_ix] / X[:, households_ix]
        population_per_households = X[:, population_ix] / X[:, households_ix]
        if self.add_bedroom_per_room:
            bedroom_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, room_per_households, population_per_households, bedroom_per_room]
        else:
            return np.c_[X, room_per_households, population_per_households]


#
#
# com = CombinedAttributesAdder(add_bedroom_per_room=False)
# housing_extra_attribs = com.transform(housing.values)

# 把数据组装到原始数据中去
# new_data = pd.DataFrame(housing_extra_attribs,
#                         columns=list(housing.columns) + ["room_per_households", "population_per_households"],
#                         index=housing.index)


# 转换流水线
num_pipeline = Pipeline([
    # 使用中位数补充缺省值
    ("imputer", SimpleImputer(strategy="median")),
    # 根据原始属性添加新属性
    ("attribs_adder", CombinedAttributesAdder()),
    # 数据归一化
    ("std_scaler", StandardScaler())
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr)

# 组合处理数字和处理文本的转换流水线
# 获取数据对应的列名
num_column = list(housing_num)
cat_column = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_column),
    ("cat", OneHotEncoder(), cat_column)
])
housing_prepared = full_pipeline.fit_transform(housing)

# 还有一种老式的数据处理方法
# 自定义转换器分类数字数据和分类数据
# class DataFrameSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, columns_name):
#         self.columns_name = columns_name
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return X[self.columns_name].values
#
#
# num_pipeline = Pipeline([
#     ("selector", DataFrameSelector(num_column)),
#     ("imputer", SimpleImputer(strategy="median")),
#     ("attribs_adder", CombinedAttributesAdder()),
#     ("sta_scaler", StandardScaler())
# ])
#
# cat_pipeline = Pipeline([
#     ("selector", DataFrameSelector(cat_column)),
#     ("cat", OneHotEncoder(sparse=False))
# ])
#
# full_pipeline = FeatureUnion([
#     ("num_pipeline", num_pipeline),
#     ("cat_pipeline", cat_pipeline)
# ])
# old_housing_prepared = full_pipeline.fit_transform(housing)
# print(old_housing_prepared)
# print(old_housing_prepared.shape)

# 比较两个转换器出来的数据是否相等
# print(np.allclose(housing_prepared, old_housing_prepared))

# 选择和训练一个模型
# 训练一个线性模型
line_reg = LinearRegression()
line_reg.fit(housing_prepared, housing_label)

# 使用几个训练集参数
some_data = housing.iloc[:5]
some_label = housing_label.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", line_reg.predict(some_data_prepared))
print("Labels:", list(some_label))

# 预测结果并检查预测的误差
housing_predict = line_reg.predict(housing_prepared)
# 均方误差
line_mse = mean_squared_error(housing_label, housing_predict)
# 均方根误差
line_rmse = np.sqrt(line_mse)
print(line_rmse)
# 平均绝对误差
mae = mean_absolute_error(housing_label, housing_predict)
print(mae)

# 使用回归决策树模型找到复杂的非线性关系
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_label)

# 预测
housing_predict = tree_reg.predict(housing_prepared)
# 评估
mae = mean_absolute_error(housing_label, housing_predict)
print(mae)
mse = mean_squared_error(housing_label, housing_predict)
rmse = np.sqrt(mse)
print(mse)
print(rmse)


# 使用交叉验证对模型进行评估
# 评估回归决策树模型
scores = cross_val_score(tree_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# display(tree_rmse_scores)

# 评估线性回归模型
# scores = cross_val_score(line_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10)
# print(scores)
# line_rmse_scores = np.sqrt(-scores)
# display(line_rmse_scores)

# 使用回归随机森林模型
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_label)

housing_predict = forest_reg.predict(housing_prepared)
mse = mean_squared_error(housing_label, housing_predict)
rmse = np.sqrt(mse)
print(rmse)

# 交叉认证随机森林模型
scores = cross_val_score(forest_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10)
forest_rmes_scores = np.sqrt(-scores)
display(forest_rmes_scores)

# 查看线性模型交叉验证得分详情
scores = cross_val_score(line_reg, housing_prepared, housing_label, scoring="neg_mean_squared_error", cv=10)
print(pd.Series(np.sqrt(-scores)).describe())

# 使用支持向量机并验证模型
# svm_reg = SVR(kernel="linear")
# svm_reg.fit(housing_prepared, housing_label)
#
# housing_predict = svm_reg.predict(housing_prepared)
# mse = mean_squared_error(housing_label, housing_predict)
# rmse = np.sqrt(mse)
# print(rmse)

# 调整模型
# 使用网格搜索调整模型参数
# 设置参数网格
forest_reg = RandomForestRegressor(random_state=42)

param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]}
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(housing_prepared, housing_label)
# 查看结果
# print(grid_search.best_params_)
# print(grid_search.best_estimator_)

# 每次循环的结果
cv_res = pd.DataFrame(grid_search.cv_results_)
print(cv_res)

# 使用随机搜索调整模型参数
param_dis = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8)
}
forest_reg = RandomForestRegressor(random_state=42)
rand_search = RandomizedSearchCV(forest_reg, param_distributions=param_dis, n_iter=10, cv=5,
                                 scoring="neg_mean_squared_error",
                                 random_state=42)
rand_search.fit(housing_prepared, housing_label)
print(rand_search.best_estimator_)

# 查看特征重要性
feature_importance = rand_search.best_estimator_.feature_importances_
num_feature = num_column
add_feature = ["room_per_households", "population_per_households", "bedroom_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_feature = list(cat_encoder.categories_[0])
all_feature = num_feature + add_feature + cat_feature

print(list(sorted(zip(feature_importance, all_feature), reverse=True)))


# 在测试集中评估你的模型
print(housing_test.info())
# 最好的模型
best_model = grid_search.best_estimator_

X_test = housing_test.drop("median_house_value", axis=1)
y_test = housing_test["median_house_value"].copy()

X_test_pre = full_pipeline.transform(X_test)
X_test_predict = best_model.predict(X_test_pre)
test_mse = mean_squared_error(y_test, X_test_predict)
print("RMSE:", np.sqrt(test_mse))


# 一个完整的管道包括准备和预测
# full_pipeline_with_predictor = Pipeline([
#     ("preparation", full_pipeline),
#     ("forest", best_model)
# ])
# full_pipeline_with_predictor.fit(housing, housing_label)
# some_data = housing.iloc[:5]
# print(housing_label.iloc[:5])
# print(full_pipeline_with_predictor.predict(some_data))

# 保存模型
# joblib.dump(full_pipeline_with_predictor, "my_model.pkl")
#
# # 加载使用模型
# my_model = joblib.load("my_model.pkl")
# my_model.fit(housing, housing_label)
# some_data = housing.iloc[:5]
# print(housing_label.iloc[:5])
# print(my_model.predict(some_data))

# 课后练习题
# 1、使用不同的超参数，如kernel="linear"（具有C超参数的多种
# 值）或kernel="rbf"（C超参数和gamma超参数的多种值），尝试一个
# 支持向量机回归器（sklearn.svm.SVR），不用担心现在不知道这些超
# 参数的含义。最好的SVR预测器是如何工作的？
svm_reg = SVR()
# 构建网格搜索的参数
grid_param = [
    {"kernel": ["linear"], "C": [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
    {"kernel": ["rbf"], "C": [1.0, 3.0, 10., 30., 100., 300., 1000.0], "gamma": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}
]

grid_search = GridSearchCV(svm_reg, grid_param, cv=5, scoring="neg_mean_squared_error", verbose=2)
grid_search.fit(housing_prepared, housing_label)

print(np.sqrt(-grid_search.best_score_))
print(grid_search.best_estimator_)


# 2.尝试用RandomizedSearchCV替换GridSearchCV
param_dis = {
    "kernel": ["linear", "rbf"],
    "C": reciprocal(20, 20000),
    "gamma": expon(scale=1.)
}

svr_reg = SVR()
random_search = RandomizedSearchCV(svr_reg, param_distributions=param_dis, cv=5, n_iter=50, verbose=2,
                                   scoring="neg_mean_squared_error",
                                   random_state=42)
random_search.fit(housing_prepared, housing_label)
print(np.sqrt(random_search.best_score_))
print(random_search.best_estimator_)

# 3.尝试在准备流水线中添加一个转换器，从而只选出最重要的属性。
# 筛选重要性方法
def index_of_top_k(arr: list, k: int):
    return np.sort(np.argpartition(arr, -k)[-k:])


# 用于选择特征的选择器
class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feat_index_ = index_of_top_k(self.feature_importances, self.k)

    def transform(self, X):
        return X.iloc[:, self.feat_index_]


# 查看选择出来的特征
k = 5
feature_importances = grid_search.best_estimator_.feature_importances_
top_5_feature_index = index_of_top_k(feature_importances, k)
print(np.array(all_feature)[top_5_feature_index])

# 构造数据处理加特征选择的管道
preparation_and_feature_selection_pipeline = Pipeline([
    ("preparation", full_pipeline),
    ("feature_selection", TopFeatureSelector(feature_importances, k))
])
preparation_and_feature_selection_pipeline.fit_transform(housing)

# 4、尝试创建一个覆盖完整的数据准备和最终预测的流水线。
# 创建一个管道 其中包含数据处理、特征选择、预测
best_estimator = grid_search.best_estimator_
last_pipeline = Pipeline([
    ("preparation", full_pipeline),
    ("feature_selection", TopFeatureSelector(feature_importances, k)),
    ("forest", best_estimator)
])
last_pipeline.fit(housing, housing_label)

# 5、使用GridSearchCV自动探索一些准备选项。
# 使用网格搜索 探索处理缺省数据时使用的策略以及筛选重要特征时的数量
# 构建搜索网格
full_pipeline.named_transformers_["cat"].handle_unknown = 'ignore'

param_grid = [{
    "preparation__num__imputer__strategy": ["mean", "median", "most_frequent"],
    "feature_selection__k": list(range(1, len(feature_importances) + 1))
}]

last_grid = GridSearchCV(last_pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", verbose=2)
last_grid.fit(housing, housing_label)

print(last_grid.best_estimator_)



