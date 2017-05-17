# -*- encoding:utf-8 -*-

from keras.layers import LSTM, GRU
from keras.preprocessing.sequence import pad_sequences
import codecs
import re
from collections import Counter
import numpy as np

with codecs.open("D:/DataForMining/sth/recipeTest.txt", encoding="utf8") as f:
    raw_data = f.read().strip().split("\n")
for line in raw_data[:5]:
    print(line)

# 数据预处理


def process_single_data(line):
    pattern = r"【菜名】([^【]*)【所属菜系】([^【]*)【特点】([^【]*).*【原料】([^【]*).*【制作过程】([^【]*).*"
    match = re.search(pattern, line)
    if match:
        return {"菜名": match.group(1),
                "菜系": match.group(2),
                "特点": match.group(3),
                "原料": match.group(4),
                "过程": match.group(5)}
    print("处理下列数据出现错误：", line)

sample = process_single_data(raw_data[1])

for k in sample:
    print(k, sample[k])

# Python3中要将map出的结果转化为list
data = list(map(process_single_data, raw_data))


def process_rnn(line):
    # 标点符号替换为空格
    line = re.sub("[。，、/…~～：；:%]", " ", line) + " "
    line = re.sub("（[^（]*）", " ", line)
    # 阿拉伯数字替换为空格
    line = re.sub("\d+\S* ", " ", line)
    return line

# print("===")
# for i in range(0, 4000, 100):
#     print(data[i]["原料"])
# print("===")
print(len(data))

data_rnn = [process_rnn(data[i]["原料"]) for i in range(0, len(data), 1)]

word_cts = Counter("".join(data_rnn))

print(word_cts)

# 去掉频数为1的
import copy
word_ctsCopy = copy.copy(word_cts)
for k, v in word_ctsCopy.items():
    if v == 1:
        word_cts.pop(k)

print(word_cts)
print(len(word_cts))

# 菜系
data_tps = [s["菜系"] for s in data]
tps_cts = Counter(data_tps)

# 两种编号
# enumerate遍历元素和下标
word_idx = {k: idx + 1 for idx, k in enumerate(word_cts)}
idx_word = {idx + 1: k for idx, k in enumerate(word_cts)}

# 菜系
tps_idx = {k: idx for idx, k in enumerate(tps_cts)}
idx_tps = {idx: k for idx, k in enumerate(tps_cts)}

print(tps_idx)

print("====================")
# x 需要预测的
# lambda 预测，输入
data_rnn_idx = list(map(lambda x: [word_idx.get(t, 0) for t in x], data_rnn))
print(data_rnn_idx)
# y 结果
data_tps_idx = np.array(list(map(lambda x: tps_idx[x], data_tps)))
print(list(map(lambda x: tps_idx[x], data_tps)))

# 整合序列 整合到同样长度
data_rnn_idx = np.array(pad_sequences(data_rnn_idx))

# 序列长度
n_seq = len(data_rnn_idx[0])
print("序列长度:", n_seq)
# 元素个数长度
n_words = len(word_idx) + 1
print("元素个数长度:", n_words)
# 预测元素个数
n_tps = len(tps_idx)
print("预测元素个数长度:", n_tps)

# model
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten

rnn_model = Sequential()
rnn_model.add(Embedding(n_words, 64, input_length=n_seq))
rnn_model.add(GRU(64))
rnn_model.add(Dense(128, activation="relu", name="feat"))
rnn_model.add(Dense(n_tps, activation="softmax"))

rnn_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

# 保存模型
rnn_model.save('D:/DataForMining/sth/my_model.model')

# 模型开始运行
hist = rnn_model.fit(data_rnn_idx, data_tps_idx,
                     nb_epoch=50, verbose=0)

# 准确率
print(hist.history["acc"][-1])

data_feat = rnn_model.predict(data_rnn_idx)

print(data_feat.shape)
print("====================")
print(data_feat)


from keras.models import Model
feat_model = Model(
    rnn_model.input, rnn_model.get_layer("feat").output)

data_feat = feat_model.predict(data_rnn_idx)
print(data_feat.shape)

# 做一个TSNE降到2维：

from sklearn.manifold import TSNE
from numpy._distributor_init import NUMPY_MKL
data_feat_vis = TSNE(n_components=2, init='pca').fit_transform(data_feat)

print(data_feat_vis.shape)
print("====================")
print(data_feat_vis)

# 可视化
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# tps_idx 是{'全部': 0, '东北菜': 1, '湘菜': 3, '鲁菜': 2}
_, ax = plt.subplots(figsize=(15, 12))
for tp, idx in tps_idx.items():
    ax.scatter(data_feat_vis[data_tps_idx == idx, 0], data_feat_vis[data_tps_idx == idx, 1],
               c=cm.Vega20.colors[idx], label=tp)

ax.set_xlim(-20, 40)
ax.set_ylim(-20, 20)
ax.set_xticks([])
ax.set_yticks([])

ax.legend(loc=0)
ltext = ax.get_legend().get_texts()
font_song = FontProperties(fname="D:/DataForMining/SIMSUN.TTC")
for t in ltext:
    t.set_font_properties(font_song)
    t.set_fontsize("xx-large")

plt.show()
