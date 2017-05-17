from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist

import numpy

model = Sequential()
model.add(Dense(784, 500, init='glorot_uniform'))  # 输入层，28*28=784
model.add(Activation('tanh'))  # 激活函数是tanh
model.add(Dropout(0.5))  # 采用50%的dropout

model.add(Dense(500, 500, init='glorot_uniform'))  # 隐层节点500个
model.add(Activation('tanh'))
model.add(Dropout(0.5))

# 输出结果是10个类别，所以维度是10
model.add(Dense(500, 10, init='glorot_uniform'))
model.add(Activation('softmax'))  # 最后一层用softmax

# 设定学习率（lr）等参数
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# 使用交叉熵作为loss函数，就是熟知的log损失函数
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, class_mode='categorical')
# 使用Keras自带的mnist工具读取数据（第一次需要联网）
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 由于输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
X_train = X_train.reshape(X_train.shape[0],
                          X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
# 这里需要把index转换成一个one hot的矩阵
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int)
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

# 开始训练，这里参数比较多。batch_size就是batch_size，nb_epoch就是最多迭代的次数， shuffle就是是否把数据随机打乱之后再进行训练
# verbose是屏显模式，官方这么说的：verbose: 0 forno logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
# 就是说0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据
# show_accuracy就是显示每次迭代后的正确率
# validation_split就是拿出百分之多少用来做交叉验证
model.fit(X_train, Y_train, batch_size=200, nb_epoch=100,
          shuffle=True, verbose=1, show_accuracy=True, validation_split=0.3)
model.evaluate(X_test, Y_test, batch_size=200, show_accuracy=True, verbose=1)
