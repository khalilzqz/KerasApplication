from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist

import numpy

model = Sequential()
model.add(Dense(784, 500, init='glorot_uniform'))  # ����㣬28*28=784
model.add(Activation('tanh'))  # �������tanh
model.add(Dropout(0.5))  # ����50%��dropout

model.add(Dense(500, 500, init='glorot_uniform'))  # ����ڵ�500��
model.add(Activation('tanh'))
model.add(Dropout(0.5))

# ��������10���������ά����10
model.add(Dense(500, 10, init='glorot_uniform'))
model.add(Activation('softmax'))  # ���һ����softmax

# �趨ѧϰ�ʣ�lr���Ȳ���
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# ʹ�ý�������Ϊloss������������֪��log��ʧ����
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, class_mode='categorical')
# ʹ��Keras�Դ���mnist���߶�ȡ���ݣ���һ����Ҫ������
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# ������������ά����(num, 28, 28)��������Ҫ�Ѻ����ά��ֱ��ƴ�������784ά
X_train = X_train.reshape(X_train.shape[0],
                          X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
# ������Ҫ��indexת����һ��one hot�ľ���
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int)
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

# ��ʼѵ������������Ƚ϶ࡣbatch_size����batch_size��nb_epoch�����������Ĵ����� shuffle�����Ƿ�������������֮���ٽ���ѵ��
# verbose������ģʽ���ٷ���ô˵�ģ�verbose: 0 forno logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
# ����˵0�ǲ����ԣ�1����ʾһ����������2��ÿ��epoch����ʾһ������
# show_accuracy������ʾÿ�ε��������ȷ��
# validation_split�����ó��ٷ�֮����������������֤
model.fit(X_train, Y_train, batch_size=200, nb_epoch=100,
          shuffle=True, verbose=1, show_accuracy=True, validation_split=0.3)
model.evaluate(X_test, Y_test, batch_size=200, show_accuracy=True, verbose=1)
