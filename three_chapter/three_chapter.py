import sys
import os
import numpy as np
import tarfile
import ssl
import re
import nltk
import urlextract
import email
import email.policy
import urllib.request
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import loadmat
from html import unescape
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.ndimage.interpolation import shift

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, \
    roc_curve, roc_auc_score, accuracy_score

# 一些基础设置
np.random.seed(42)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 获取MNIST数据（为了节省时间和方便在无网的情况下运作，我直接下载的了数据集）
# mnist = fetch_openml("mnist_784", version=1, as_frame=False)
# print(mnist.keys())
mnist = loadmat("./data/mnist-original.mat")
X, y = mnist["data"], mnist["label"]

# 对数据进行转换，打乱顺序
all_data = np.vstack((X, y))
all_data = all_data.T
np.random.shuffle(all_data)
X = all_data[:, range(784)]
y = all_data[:, 784]
y = y.astype(np.uint8)


# 显示一张图片
def show_img(data):
    """
    显示一张图片
    :param data:
    :return:
    """
    data = data.reshape(28, 28)
    plt.imshow(data, cmap=mpl.cm.binary, interpolation="nearest")
    plt.axis("off")


one_digit = X[0]
plt.show()

# show_img(one_digit)
# print(y[0])

# 显示一百张图片
# def show_digit(data, num_per_row=10):
#     """
#     显示很多图片
#     :param data:
#     :param num_per_row:
#     :return:
#     """
#     size = 28
#     images_len = len(data)
#     num_per_row = min(images_len, num_per_row)
#     row_num = (images_len - 1) // num_per_row + 1
#
#     # 用空白填充空余
#     blank = row_num * row_num - images_len
#     all_data = np.concatenate([data, np.zeros((blank, size * size))], axis=0)
#     img_grid = all_data.reshape((row_num, num_per_row, size, size))
#     big_image = img_grid.transpose(0, 2, 1, 3).reshape(row_num * size, num_per_row * size)
#     plt.imshow(big_image, cmap=mpl.cm.binary)
#     plt.axis("off")


# top_100_img = X[10000:10100]
# show_digit(top_100_img)
# plt.show()

# 生成测试集和训练集
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 训练一个二元分类器区分是5和不是5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# print(y_train_5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([one_digit]))

# 交叉验证
cvs = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(cvs)

# 使用StratifiedKFold自定义交叉验证
sk_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in sk_fold.split(X_train, y_train_5):
    clone_sgd = clone(sgd_clf)
    fold_X_train = X_train[train_index]
    fold_y_train = y_train_5[train_index]
    fold_X_test = X_train[test_index]
    fold_y_test = y_train_5[test_index]

    clone_sgd.fit(fold_X_train, fold_y_train)
    pre_y = clone_sgd.predict(fold_X_test)
    num_true = sum(np.array(fold_y_test) == np.array(pre_y))
    print(num_true / len(pre_y))


# 自定义一个分类器分类器的目的是把所有数据都分类为不是5
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


print(sum(y_train_5) / len(y_train_5))
print(len(X_train))
# 交叉验证这个全部为非5的分类器的性能
never_5 = Never5Classifier()
cvs = cross_val_score(never_5, X_train, y_train_5, cv=3, scoring="accuracy")
print(cvs)

# 混淆矩阵
# 随机梯度下降分类器的混淆矩阵
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

# 完美的混淆矩阵
perfect_predictions = y_train_5
cm = confusion_matrix(y_train_5, perfect_predictions)
print(cm)

# 精度与召回率
precision = precision_score(y_train_5, y_train_pred)
print(precision)
recall = recall_score(y_train_5, y_train_pred)
print(recall)
# f1分数，结合了精度和召回率的值
f1 = f1_score(y_train_5, y_train_pred)
print(f1)

# 精度和召回率之间的权衡
# 使用decision_function查看模型用于预测的决策分数
# y_score = sgd_clf.decision_function([one_digit])
# print(y_score)
# 设置并调整阈值
# threshold = -10000
# one_digit_pre = (y_score > threshold)
# print(one_digit_pre)
#
# threshold = 0
# one_digit_pre = (y_score > threshold)
# print(one_digit_pre)

# 绘制精度召回率曲线
y_score = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precision, recall, threshold = precision_recall_curve(y_train_5, y_score)


def show_precision_recall_vs_threshold(precision, recall, threshold):
    """
    显示精度召回率和阈值之间的关系
    :param precision:
    :param recall:
    :param threshold:
    :return:
    """
    plt.plot(threshold, precision[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(threshold, recall[:-1], "g--", label="Precision", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.axis([-10000, 10000, 0, 1])
    plt.grid(True)


# 计算精度达到90时的阈值和召回率
first_max_index = np.argmax(precision >= 0.9)
precision_90_recall = recall[first_max_index]
precision_90_threshold = threshold[first_max_index]

show_precision_recall_vs_threshold(precision, recall, threshold)
# 添加辅助线
plt.plot([precision_90_threshold, precision_90_threshold], [0., 0.9], "r:")
plt.plot([-10000, precision_90_threshold], [0.9, 0.9], "r:")
plt.plot([-10000, precision_90_threshold], [precision_90_recall, precision_90_recall], "r:")
# 添加交点
plt.plot([precision_90_threshold], [0.9], "ro")
plt.plot([precision_90_threshold], [precision_90_recall], "ro")
plt.show()

# print((y_train_pred == (y_score > 0)).all())


def show_recall_precision(recall, precision):
    """
    绘制召回率与精度的关系
    :param recall:
    :param precision:
    :return:
    """
    plt.plot(recall, precision, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)


plt.figure(figsize=(8, 6))
show_recall_precision(recall, precision)
# 计算精度到达90时的召回率
precision_90_recall = recall[np.argmax(precision >= 0.9)]
# 绘制辅助线
plt.plot([precision_90_recall, precision_90_recall], [0, 0.9], "r:")
plt.plot([0, precision_90_recall], [0.9, 0.9], "r:")
# 绘制交点
plt.plot([precision_90_recall], [0.9], "ro")
plt.show()

# 使用精度大于90的阈值来查看精度和召回率
precision_90_threshold = threshold[np.argmax(precision >= 0.9)]

y_train_90 = (y_score >= precision_90_threshold)

recall_90 = recall_score(y_train_5, y_train_90)
precision_90 = precision_score(y_train_5, y_train_90)
print(recall_90)
print(precision_90)

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_train_5, y_score)


# 绘制roc曲线
def show_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("FPR", fontsize=16)
    plt.ylabel("TPR", fontsize=16)
    plt.grid(True)


plt.figure(figsize=(8, 6))
show_roc_curve(fpr, tpr)
# 查看到精度到达90的召回率（tpr）时fpr的值为多少
fpr_precision_90 = fpr[np.argmax(tpr >= precision_90_recall)]
print(fpr_precision_90)
plt.plot([fpr_precision_90, fpr_precision_90], [0, precision_90_recall], ":r")
plt.plot([0, fpr_precision_90], [precision_90_recall, precision_90_recall], ":r")
plt.plot([fpr_precision_90], [precision_90_recall], "ro")
plt.show()

# 查看roc曲线的auc
auc = roc_auc_score(y_train_5, y_score)
print(auc)

# 训练一个随机森林分类器模型
forest_fcl = RandomForestClassifier(n_estimators=100, random_state=42)
forest_fcl.fit(X_train, y_train_5)
# 交叉验证
y_proba_forest = cross_val_predict(forest_fcl, X_train, y_train_5, cv=3, method="predict_proba")
y_score_forest = y_proba_forest[:, 1]

# 绘制roc曲线
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_score_forest)
tpr_forest_90 = tpr_forest[np.argmax(fpr_forest >= fpr_precision_90)]

plt.figure(figsize=(8, 6))
plt.plot(fpr_forest, tpr_forest, label="RANDOM FOREST")
show_roc_curve(fpr, tpr, label="SGD")
# 绘制辅助线
plt.plot([fpr_precision_90, fpr_precision_90], [0., precision_90_recall], "r:")
plt.plot([0., fpr_precision_90], [precision_90_recall, precision_90_recall], "r:")
plt.plot([fpr_precision_90, fpr_precision_90], [0., tpr_forest_90], "r:")
plt.plot([fpr_precision_90], [precision_90_recall], "ro")
plt.plot([fpr_precision_90], [tpr_forest_90], "ro")
plt.legend(loc="lower right", fontsize=16)
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.show()

# 计算auc、召回率和精度
auc = roc_auc_score(y_train_5, y_score_forest)
print(auc)

predict_forest = cross_val_predict(forest_fcl, X_train, y_train_5, cv=3)
print(predict_forest)
recall_forest = recall_score(y_train_5, predict_forest)
print(recall_forest)

precision_forest = precision_score(y_train_5, predict_forest)
print(precision_forest)

# 多类分类器
svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X[:1000], y_train[:1000])
svm_predict = svm_clf.predict([one_digit])
# print(svm_predict)

svm_dec = svm_clf.decision_function([one_digit])
print(svm_dec)

print(svm_clf.classes_)

print(svm_clf.classes_[7])

# 训练一个一对剩余的模型
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X[:1000], y_train[:1000])
ovr_predict = ovr_clf.predict([one_digit])
print(ovr_predict)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X[:1000], y_train[:1000])
sgd_predict = sgd_clf.predict([one_digit])
print(sgd_predict)

random_forest_clf = RandomForestClassifier(random_state=42)
random_forest_clf.fit(X[:1000], y_train[:1000])
random_forest_predict = random_forest_clf.predict([one_digit])
print(random_forest_predict)

# 验证归一化是否能够提高准确度
sgd_score = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print(sgd_score)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
sgd_scaled_score = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3)
print(sgd_scaled_score)

# 错误分析
# 混淆矩阵
sgd_clf = SGDClassifier(random_state=42)
y_train_predict = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
sgd_cm = confusion_matrix(y_train, y_train_predict)

def show_confusion_matrix(matrix):
    """
    手动绘制混淆矩阵
    :param matrix:
    :return:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


# show_confusion_matrix(sdg_cm)
# plt.show()

# sklearn绘制混淆矩阵
plt.matshow(sgd_cm, cmap=plt.cm.gray)
plt.show()

# 计算每一类图片的数量
row_nums = sgd_cm.sum(axis=1, keepdims=True)
# 考虑每个数字的图片数量不同，考虑到数据的准确性需要对混淆矩阵的数值除以他的数量
norm_sgd_cm = sgd_cm / row_nums
# 用0填充对角线
np.fill_diagonal(norm_sgd_cm, 0)
plt.matshow(norm_sgd_cm, cmap=plt.cm.gray)
plt.show()

# 检查3和5的混淆
num_3, num_5 = 3, 5
X_33 = X_train[(y_train == num_3) & (y_train_predict == num_3)]
X_35 = X_train[(y_train == num_3) & (y_train_predict == num_5)]
X_53 = X_train[(y_train == num_5) & (y_train_predict == num_3)]
X_55 = X_train[(y_train == num_5) & (y_train_predict == num_5)]

# 绘制3和5的错误分类图，左上角为是正确分类为3的，右上角为错误分类为3的
# 左下角为错误分类为5的3的图片，右下角为正确分类为5的图片
# plt.figure(figsize=(8, 8))
# plt.subplot(221)
# show_digit(X_33[:25], num_per_row=5)
# plt.subplot(222)
# show_digit(X_35[:25], num_per_row=5)
# plt.subplot(223)
# show_digit(X_53[:25], num_per_row=5)
# plt.subplot(224)
# show_digit(X_55[:25], num_per_row=5)
# plt.show()


# 多标签分类
y_train_large = (y_train >= 7)
y_train_even = (y_train % 2 == 0)
y_train_large_odd = np.c_[y_train_large, y_train_even]
# 使用K临近算法训练模型
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train_large_odd)
knn_predict = knn_clf.predict([one_digit])
print(knn_predict)


# 多输出分类
# 对图片添加噪音
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_noise = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_noise = X_test + noise
y_train_noise = X_train


# 显示添加噪音和不添加噪音的区别
plt.figure(figsize=(8, 4))
plt.subplot(121)
show_img(X_test[0])
plt.subplot(122)
show_img(X_test_noise[0])
plt.show()

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_noise, y_train_noise)
clean_digit = knn_clf.predict([X_test_noise[0]])
show_img(clean_digit)
plt.show()


# 课后练习
# 1、为MNIST数据集构建一个分类器，并在测试集上达成超过97%的
# 准确率。提示：KNeighborsClassifier对这个任务非常有效，你只需
# 要找到合适的超参数值即可（试试对weights和n_neighbors这两个超
# 参数进行网格搜索）。
# 构建网格参数
grid_param = [{"weights": ["uniform", "distance"], "n_neighbors": [3, 4, 5]}]
# 开始网格搜索
knn_clf = KNeighborsClassifier()
grid_cv = GridSearchCV(knn_clf, grid_param, cv=5, verbose=3)
grid_cv.fit(X_train, y_train)

print(grid_cv.best_estimator_)
print(grid_cv.best_score_)
knn_test_predict = grid_cv.predict(X_test)
knn_score = accuracy_score(y_test, knn_test_predict)
print(knn_score)


# 2、写一个可以将MNIST图片向任意方向（上、下、左、右）移动一
# 个像素的功能[1]。然后对训练集中的每张图片，创建四个位移后的副
# 本（每个方向一个），添加到训练集。最后，在这个扩展过的训练集
# 上训练模型，测量其在测试集上的准确率。你应该能注意到，模型的
# 表现甚至变得更好了！这种人工扩展训练集的技术称为数据增广或训
# 练集扩展。

# 设置图片偏移
def shift_image(image, dx, dy):
    image = image.reshape(28, 28)
    shifted_image = shift(image, [dx, dy], cval=0)
    return shifted_image.reshape([-1])

shifted_image_down = shift_image(one_digit, -5, 0)
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.title("Original", fontsize=16)
plt.imshow(one_digit.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(122)
plt.title("Shift", fontsize=16)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()

# 循环偏移图片
X_train_augmented = [image for image in X_train]
y_train_augmented = [image for image in y_train]

for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# 打乱顺序
per_index = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[per_index]
y_train_augmented = y_train_augmented[per_index]

# 训练模型
knn_clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn_clf.fit(X_train_augmented, y_train_augmented)
knn_augmented_predict = knn_clf.predict(X_test)
knn_augmented_score = accuracy_score(y_test, knn_augmented_predict)
print(knn_augmented_score)

# 3、Kaggle上非常棒的起点：处理泰坦尼克（Titanic）数据集。
# 有示例

# 4、创建一个垃圾邮件分类器
# 下载数据
ssl._create_default_https_context = ssl._create_unverified_context
root_path = "http://spamassassin.apache.org/old/publiccorpus/"
ham_url = root_path + "20030228_easy_ham.tar.bz2"
spam_url = root_path + "20030228_spam.tar.bz2"
save_path = "data"

def download_data(ham=ham_url, spam=spam_url, save=save_path):
    if not os.path.isdir(save):
        os.makedirs(save)
    for file_name, file_url in (("ham.tar.bz2", ham), ("spam.tar.bz2", spam)):
        path = os.path.join(save, file_name)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(file_url, path)

        # 解压
        tar_file = tarfile.open(path)
        tar_file.extractall(save)
        tar_file.close()


download_data()

# 加载数据
easy_ham_path = os.path.join(save_path, "easy_ham")
spam_path = os.path.join(save_path, "spam")

ham_file_name = [file_name for file_name in sorted(os.listdir(easy_ham_path)) if len(file_name) > 20]
spam_file_name = [file_name for file_name in sorted(os.listdir(spam_path)) if len(file_name) > 20]


# 解析邮件
def load_email(is_spam, file_name, email_path=save_path):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(email_path, directory, file_name), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


ham_email = [load_email(False, name) for name in ham_file_name]
spam_email = [load_email(True, name) for name in spam_file_name]


# print(ham_email[0].get_content().strip())
# print(spam_email[0].get_content().strip())

# 查看邮件的种类数量
def get_email_structure(email):
    """
    检查邮件的类型
    :param email:
    :return:
    """
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


# def structures_counter(emails):
#     """
#     用于计算每个种类有多少的数量
#     :param emails:
#     :return:
#     """
#     structures = Counter()
#     for email in emails:
#         structure = get_email_structure(email)
#         structures[structure] += 1
#     return structures
#
#
# ham_type_num = structures_counter(ham_email).most_common()
# spam_type_num = structures_counter(spam_email).most_common()
#
# print(ham_type_num)
# print(spam_type_num)

# 查看邮件的标题
# for header, value in spam_email[0].items():
#     print(header, value)
#
# print(spam_email[0]["Subject"])

# 把邮件数据分类训练集和测试集
X = np.array(ham_email + spam_email, dtype=object)
y = np.array([0] * len(ham_email) + [1] * len(spam_email))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 把html的邮件转化为纯文本
def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


# 获取所有垃圾邮件中类型为HTML的邮件
# spam_html = [email for email in X_train[y_train == 1] if get_email_structure(email) == "text/html"]


# print(html_to_plain_text(spam_html[7].get_content())[:1000], "...")


# 封装一个方法使所有的邮件返回值都是纯文本
def email_to_text(email):
    html = None
    for part in email.walk():
        c_type = part.get_content_type()
        if c_type not in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())
        if c_type == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


# print(email_to_text(spam_html[7])[:100])

# 使用nltk提取文章词干
# 测试nltk的使用
# stemmer = nltk.PorterStemmer()
# for work in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
#     try:
#         print(work, "==>", stemmer.stem(work))
#     except ImportError:
#         print("Error: stemming requires the NLTK module.")
#         stemmer = None

# 使用urlextract处理文章url
# extract = urlextract.URLExtract()
# print(extract.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))

# 创建用于解析url的类
url_extract = urlextract.URLExtract()
stemmer = nltk.PorterStemmer()


# 编写一个把所有都整合到一起的转换器
class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, split_header=True, lower_case=True, remove_punctuation=True, replace_urls=True,
                 replace_numbers=True, stemming=True):
        self.split_header = split_header
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transform = []
        for email in X:
            # 把所有格式的邮件转成字符串
            text = email_to_text(email) or ""
            # 文本转小写
            if self.lower_case:
                text = text.lower()
            # 文本中所有URL替换成”URL“
            if self.replace_urls and url_extract is not None:
                urls = list(set(url_extract.find_urls(text)))
                # 把URL根据程度排序
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text.replace(url, "URL")
            # 替换文本中的数字
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            # 删除表单符号
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            # 计算每个词出现的频率
            word_count = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_count.items():
                    stemmer_word = stemmer.stem(word)
                    stemmed_word_counts[stemmer_word] += 1
                word_count = stemmed_word_counts
            X_transform.append(word_count)
        return np.array(X_transform)


# 测试一下转换器
X_few = X_train[:3]
X_few_word_content_count = EmailToWordCounterTransformer().fit_transform(X_few)


# 再定义一个转换器把统计到的词汇转换成词汇表
class WordContentToVectorTransFormer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


# vocab_transformer = WordContentToVectorTransFormer(vocabulary_size=10)
# X_few_vectors = vocab_transformer.fit_transform(X_few_word_content_count)
# print(X_few_vectors)
# print(X_few_vectors.toarray())
# print(X_few_vectors.vocabulary_)
preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordContentToVectorTransFormer())
])

X_train_transformer = preprocess_pipeline.fit_transform(X_train)

# 训练一个逻辑回归模型并计算分数
# log_reg = LogisticRegression(max_iter=1000, random_state=42)
# log_reg_score = cross_val_score(log_reg, X_train_transformer, y_train, cv=5, verbose=3)
# print(log_reg_score.mean())

# 训练一个逻辑分类模型并计算精度和召回率
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_transformer, y_train)
y_predict = log_reg.predict(X_train_transformer)
print(recall_score(y_train, y_predict))
print(precision_score(y_train, y_predict))
