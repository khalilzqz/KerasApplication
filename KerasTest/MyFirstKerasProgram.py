# coding=gbk
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
import numpy as np


def readFile(path):
    # ���ļ���ע��·����
    f = open(path)
    # ���н��д���
    first_ele = True
    for data in f.readlines():
        # ȥ��ÿ�еĻ��з���"\n"
        data = data.strip('\n')
        # ���� �ո���зָ
        nums = data.split(",")
        # ��ӵ� matrix �С�
        if first_ele:
            # ���ַ���ת��Ϊ��������
            nums = [int(x) for x in nums]
            # ���뵽 matrix �� ��
            matrix = np.array(nums)
            first_ele = False
        else:
            nums = [int(x) for x in nums]
            matrix = np.c_[matrix, nums]
    data = np.delete(matrix, 0, 0)
    label = matrix[0]
    matrix = data.transpose()
    f.close()
    return data, label

# ��������
data, label = readFile("D:\DataForMining\Classify\mnt.txt")
print(data.shape[0], ' samples')

# coding=gbk
# labelΪ0~9��10�����kerasҪ���ʽΪbinary class matrices,ת��һ�£�ֱ�ӵ���keras�ṩ���������
label = np_utils.to_categorical(label, 10)

###############
# ��ʼ����CNNģ��
###############

# ����һ��model
model = Sequential()

# ��һ������㣬4������ˣ�ÿ������˴�С5*5��1��ʾ�����ͼƬ��ͨ��,�Ҷ�ͼΪ1ͨ����
# border_mode������valid����full�����忴����˵����http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
# �������tanh
# �㻹������model.add(Activation('tanh'))�����dropout�ļ���: model.add(Dropout(0.5))
model.add(
    Convolution2D(16, 4, 4, border_mode='valid', input_shape=data.shape[-3:]))
model.add(Activation('tanh'))

# �ڶ�������㣬8������ˣ�ÿ������˴�С3*3��4��ʾ���������ͼ������������һ��ľ���˸���
# �������tanh
# ����maxpooling��poolsizeΪ(2,2)
model.add(Convolution2D(32, 4, 4, border_mode='valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(poolsize=(2, 2)))

# ����������㣬16������ˣ�ÿ������˴�С3*3
# �������tanh
# ����maxpooling��poolsizeΪ(2,2)
model.add(Convolution2D(32, 4, 4, border_mode='valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(poolsize=(2, 2)))

# ȫ���Ӳ㣬�Ƚ�ǰһ������Ķ�ά����ͼflattenΪһά�ġ�
# Dense�������ز㡣16������һ�����������ͼ������4�Ǹ���ÿ��������������ģ�(28-5+1)�õ�24,(24-3+1)/2�õ�11��(11-3+1)/2�õ�4
# ȫ������128����Ԫ�ڵ�,��ʼ����ʽΪnormal
model.add(Flatten())
model.add(Dense(128, init='normal'))
model.add(Activation('tanh'))

# Softmax���࣬�����10���
model.add(Dense(128, 10, init='normal'))
model.add(Activation('softmax'))

#############
# ��ʼѵ��ģ��
##############
# ʹ��SGD + momentum
# model.compile��Ĳ���loss������ʧ����(Ŀ�꺯��)
sgd = SGD(l2=0.0, lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

# ����fit����������һ��ѵ������. ѵ����epoch����Ϊ10��batch_sizeΪ100��
# ���ݾ����������shuffle=True��verbose=1��ѵ���������������Ϣ��0��1��2���ַ�ʽ�����ԣ��޹ؽ�Ҫ��show_accuracy=True��ѵ��ʱÿһ��epoch�����accuracy��
# validation_split=0.2����20%��������Ϊ��֤����
model.fit(data, label, batch_size=100, nb_epoch=10, shuffle=True,
          verbose=1, show_accuracy=True, validation_split=0.2)

# fit�����ڴﵽ�趨��nb_epochʱ�����������Զ��ر�����Ч����õ�model,֮������Ե���model.evaluate()�����Բ������ݽ��в��ԣ�
# ����model.predict_classes,model.predict_proba�ȷ���,�����뿴�ĵ���
