# -*- encoding:utf-8 -*-
'''
Created on 2017年4月5日
https://github.com/lijin-THU/play-with-machine-learning
@author: zhouqizhao
'''
from __future__ import unicode_literals
import codecs
from collections import Counter
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import re

with codecs.open("D:/DataForMining/sth/recRecipePredictiont", encoding="utf8") as f:
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

# 基本作图
# font_song = FontProperties(fname="D:/DataForMining/SIMSUN.TTC")
# plt.title("中文", fontproperties=font_song)
# plt.show()
#
# 排序
# data_tps = [s["菜系"] for s in data]
#
# tps_cts = Counter(data_tps)
#
# for k, v in tps_cts.items():
#     print(k, v)
#
#
# pie图
# _, ax = plt.subplots(figsize=(8, 8))
# ax.pie([v for k, v in tps_cts.items()], labels=[
#        k for k, v in tps_cts.items()], colors=cm.Vega20.colors)
# ax.axis("equal")
# for t in ax.texts:
#     t.set_font_properties(font_song)
#
# plt.show()

# 基本处理


def process_single_ingredient(line):
    # 标点符号替换为空格
    line = re.sub("[。，、/…~～：；:%]", " ", line) + " "
    line = re.sub("（[^（]*）", " ", line)
    # 阿拉伯数字替换为空格
    line = re.sub("\d+\S* ", " ", line)
    # 汉语数字替换为空格
    line = re.sub("[一二两三四五六七八九十几半]+\S* ", " ", line)
    # 关键词替换为空格
    for s in ["少许", "适量", "或", "等", "重", "约",
              "各", "原料", "调味料", "主料", "辅料", "调料", "用料", "和", "及"]:
        line = re.sub(s, " ", line)
    # 字母替换为空格
    line = re.sub(" \w+ ", " ", line)
    line = re.sub(" +", " ", line).strip()
    return line

print("=====")
for i in range(0, 4000, 400):
    print(process_single_ingredient(data[i]["原料"]))

data_ings = map(lambda x: process_single_ingredient(x["原料"]), data)
data_ings_all = " ".join(data_ings).split()

ings_cts = Counter(data_ings_all)

for k, v in ings_cts.items():
    if v > 200:
        print(k, v)

# 有条件的统计
ings_ctsC = []
print(len(ings_cts))
for k in ings_cts.keys():
    if ings_cts[k] > 5:
        #         print(k, ings_cts[k])
        ings_ctsC.append(k)
print(len(ings_ctsC))


# 将他们编号
word_idx = {k: idx + 1 for idx, k in enumerate(ings_ctsC)}
idx_word = {idx + 1: k for idx, k in enumerate(ings_ctsC)}

print(word_idx)
print(idx_word)
# 将数据id化
data_ings_idx = map(lambda x: [word_idx.get(t, 0)
                               for t in x.split()], ings_ctsC)

print('========')


# TensorFlow
import keras.backend as K

if K.backend() == "tensorflow":
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)
    K.set_session(session)
# 模型其实很简单，我们先构造一个N×D的Embedding矩阵，其中N是我的食材数目，D是我需要的食材向量的维度，那么N种食材就对应与这N个D维向量了。

from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
from keras.regularizers import l2

n_words = len(word_idx) + 1
model = Sequential()
model.add(
    Embedding(n_words, 60, input_length=1, activity_regularizer=l2(0.01)))
model.add(Flatten())
model.add(Dense(n_words, activation="sigmoid", activity_regularizer=l2(0.01)))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

data_x = []
data_y = []

for ings_idxs in data_ings_idx:
    for idx in set(ings_idxs):
        y = np.zeros(n_words)
        data_x.append(idx)
        y[list(set(ings_idxs) - set([idx]))] = 1
        data_y.append(y)

data_x = np.array(data_x)
data_y = np.array(data_y)

rd_idx = np.arange(len(data_x))
np.random.shuffle(rd_idx)

print(data_x)
print(data_y)
print('========')
# 训练模型
hist = model.fit(
    data_x[rd_idx], data_y[rd_idx], validation_split=0.1, verbose=0,
    nb_epoch=1)


# 训练结束
emb = model.get_weights()[0]
print(emb.shape)

# 算个距离度量，并看看一些跟食材最接近的都是什么：

from scipy.spatial import distance

dist = distance.squareform(distance.pdist(emb, "cosine"))

# print(dist[word_idx["西红柿"]])
print(idx_word[np.argsort(dist[word_idx["西红柿"]])[1]])
