# -*- encoding:utf-8 -*-
# from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model, save_model
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import os
import sys
import pandas as pd
import numpy as np


def NN_model_train(self, trainX, trainY, testX, testY, model_save_path):
    """
    :param trainX: training data set
    :param trainY: expect value of training data
    :param testX: test data set
    :param testY: expect value of test data
    :param model_save_path: h5 file to store the trained model
    :param override: override existing models
    :return: model after training
    """
    input_dim = trainX[0].shape[1]
    output_dim = trainY.shape[1]
    # print predefined parameters of current model:
    model = Sequential()
    # applying a LSTM layer with x dim output and y dim input. Use dropout
    # parameter to avoid overfit
    model.add(LSTM(output_dim=self.lstm_output_dim,
                   input_dim=input_dim,
                   activation=self.activation_lstm,
                   dropout_U=self.drop_out,
                   return_sequences=True))
    for i in range(self.lstm_layer - 2):
        model.add(LSTM(output_dim=self.lstm_output_dim,
                       activation=self.activation_lstm,
                       dropout_U=self.drop_out,
                       return_sequences=True))
    # return sequences should be False to avoid dim error when concatenating
    # with dense layer
    model.add(LSTM(output_dim=self.lstm_output_dim,
                   activation=self.activation_lstm, dropout_U=self.drop_out))
    # applying a full connected NN to accept output from LSTM layer
    for i in range(self.dense_layer - 1):
        model.add(
            Dense(output_dim=self.lstm_output_dim, activation=self.activation_dense))
        model.add(Dropout(self.drop_out))
    model.add(Dense(output_dim=output_dim, activation=self.activation_last))
    # configure the learning process
    model.compile(
        loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
    # train the model with fixed number of epoches
    model.fit(x=trainX, y=trainY, nb_epoch=self.nb_epoch,
              batch_size=self.batch_size, validation_data=(testX, testY))
    score = model.evaluate(trainX, trainY, self.batch_size)
    print("Model evaluation: {}".format(score))
    # store model to json file
    save_model(model, model_save_path)

####################################################
from sklearn import linear_model
# linear


def get_data(file_name):
    data = pd.read_csv(file_name)
    X = []
    Y = []
    for single_square_feet, single_price_value in zip(data['numid'], data['number']):
        X.append([float(single_square_feet)])
        Y.append([float(single_price_value)])
    return X, Y


def linear_model_main(X, Y, predict_value):
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions


def get_predicted_num(inputfile, num):
    X, Y = get_data(inputfile)
    predictvalue = 51
    result = linear_model_main(X, Y, predictvalue)

    print("num" + str(num) + "intercept value", result['intercept'])
    print("num" + str(num) + "coefficient", result['coefficient'])
    print("num" + str(num) + "predicted_value", result['predicted_value'])

get_predicted_num("num.txt", 6)
