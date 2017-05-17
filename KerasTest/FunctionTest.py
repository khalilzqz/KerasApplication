# -*- encoding:utf-8 -*-

from keras.layers import LSTM, GRU, merge
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

# 整合序列
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


##############################
# Function模型
from keras.layers import Input, Dense
from keras.layers import Embedding, Flatten
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(81,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=239, input_length=81)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)


auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)


auxiliary_input = Input(shape=(187, 81), name='aux_input')
feat = merge([lstm_out, auxiliary_input], mode='concat')

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)


model = Model([main_input, auxiliary_input],
              [main_output, auxiliary_output])


model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])
print("fit&compile############")

# model.fit([data_rnn_idx, data_rnn_idx], [data_tps_idx, data_tps_idx],
#           epochs=50, batch_size=32)


model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy',
                    'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': data_rnn_idx, 'aux_input': data_rnn_idx},
          {'main_output': data_tps_idx, 'aux_output': data_tps_idx},
          nb_epoch=50, batch_size=32)
