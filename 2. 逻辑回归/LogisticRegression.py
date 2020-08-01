import numpy as np
from sklearn.utils import shuffle

class LogisticRegression():
    def __init__(self):
        pass


    def sigmoid(self, x):
        z = 1 / (1 + np.exp(-x))
        return z


    def initialize_params(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b


    def logistic_loss(self, X, y, w, b):
        num_train = X.shape[0]
        # num_feature = X.shape[1]

        y_hat = self.sigmoid(np.dot(X, w) + b)
        cost = -1 / num_train * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        dw = np.dot(X.T, (y_hat - y)) / num_train
        db = np.sum(y_hat - y) / num_train
        cost = np.squeeze(cost)

        return y_hat, cost, dw, db


    def logistic_train(self, X, y, learning_rate, epochs):
        w, b = self.initialize_params(X.shape[1])
        loss_list = []

        for i in range(1, epochs):
            # 计算当前次的模型计算结果、损失和参数梯度
            y_hat, loss, dw, db = self.logistic_loss(X, y, w, b)
            
            # 基于梯度下降的参数更新过程
            w -= learning_rate * dw
            b -= learning_rate * db

            # 打印迭代次数和损失
            if i % 100 == 0:
                loss_list.append(loss)
                print('epoch %d loss %f' % (i, loss))

            # 保存参数
        params = {
            'w' : w,
            'b' : b
        }

        # 保存梯度
        grads = {
            'dw' : dw,
            'db' : db
        }

        return loss_list, params, grads


    def predict(self, X, params):
        w = params['w']
        b = params['b']

        y_pred = self.sigmoid(np.dot(X, w) + b)
        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred


    def accuracy(self, y_test, y_pred):
        correct_count = 0
        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                correct_count += 1

        accuracy_score = correct_count / len(y_test)
        return accuracy_score

