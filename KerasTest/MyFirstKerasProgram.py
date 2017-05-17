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
    # 打开文件（注意路径）
    f = open(path)
    # 逐行进行处理
    first_ele = True
    for data in f.readlines():
        # 去掉每行的换行符，"\n"
        data = data.strip('\n')
        # 按照 空格进行分割。
        nums = data.split(",")
        # 添加到 matrix 中。
        if first_ele:
            # 将字符串转化为整型数据
            nums = [int(x) for x in nums]
            # 加入到 matrix 中 。
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

# 加载数据
data, label = readFile("D:\DataForMining\Classify\mnt.txt")
print(data.shape[0], ' samples')

# coding=gbk
# label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
label = np_utils.to_categorical(label, 10)

###############
# 开始建立CNN模型
###############

# 生成一个model
model = Sequential()

# 第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
# border_mode可以是valid或者full，具体看这里说明：http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
# 激活函数用tanh
# 你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))
model.add(
    Convolution2D(16, 4, 4, border_mode='valid', input_shape=data.shape[-3:]))
model.add(Activation('tanh'))

# 第二个卷积层，8个卷积核，每个卷积核大小3*3。4表示输入的特征图个数，等于上一层的卷积核个数
# 激活函数用tanh
# 采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(32, 4, 4, border_mode='valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(poolsize=(2, 2)))

# 第三个卷积层，16个卷积核，每个卷积核大小3*3
# 激活函数用tanh
# 采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(32, 4, 4, border_mode='valid'))
model.add(Activation('tanh'))
model.add(MaxPooling2D(poolsize=(2, 2)))

# 全连接层，先将前一层输出的二维特征图flatten为一维的。
# Dense就是隐藏层。16就是上一层输出的特征图个数。4是根据每个卷积层计算出来的：(28-5+1)得到24,(24-3+1)/2得到11，(11-3+1)/2得到4
# 全连接有128个神经元节点,初始化方式为normal
model.add(Flatten())
model.add(Dense(128, init='normal'))
model.add(Activation('tanh'))

# Softmax分类，输出是10类别
model.add(Dense(128, 10, init='normal'))
model.add(Activation('softmax'))

#############
# 开始训练模型
##############
# 使用SGD + momentum
# model.compile里的参数loss就是损失函数(目标函数)
sgd = SGD(l2=0.0, lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
    loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

# 调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
# 数据经过随机打乱shuffle=True。verbose=1，训练过程中输出的信息，0、1、2三种方式都可以，无关紧要。show_accuracy=True，训练时每一个epoch都输出accuracy。
# validation_split=0.2，将20%的数据作为验证集。
model.fit(data, label, batch_size=100, nb_epoch=10, shuffle=True,
          verbose=1, show_accuracy=True, validation_split=0.2)

# fit方法在达到设定的nb_epoch时结束，并且自动地保存了效果最好的model,之后你可以调用model.evaluate()方法对测试数据进行测试，
# 还有model.predict_classes,model.predict_proba等方法,具体请看文档。
