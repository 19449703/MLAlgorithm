import numpy as np
from sklearn.utils import shuffle

class LinearRegression():
    def __init__(self):
        pass

    def linear_loss(self, X, y, w, b):
        num_train = X.shape[0]
        # num_feature = X.shape[1]

        # 模型公式
        y_hat = np.dot(X, w) + b
        # 损失函数
        loss = np.sum((y_hat - y)**2) / num_train
        # 参数的偏导
        dw = np.dot(X.T, y_hat - y)
        db = np.sum(y_hat - y) / num_train

        return y_hat, loss, dw, db


    def initialize_params(self, dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b


    def linear_train(self, X, y, learning_rate, epochs):
        w, b = self.initialize_params(X.shape[1])
        loss_list = []

        for i in range(1, epochs):
            # 计算当前预测值、损失和参数偏导
            y_hat, loss, dw, db = self.linear_loss(X, y, w, b)
            loss_list.append(loss)

            # 基于梯度下降的参数更新过程
            w -= learning_rate * dw
            b -= learning_rate * db

            # 打印迭代次数和损失
            if i % 10000 == 0:
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

        y_pred = np.dot(X, w) + b
        return y_pred


    def linear_cross_validation(self, data, k, randomize=True):
        if randomize:
            data = list(data)
            data = shuffle(data)

        slices = [data[i::k] for i in range(k)]
        for i in range(k):
            validation = slices[i]
            train = [data for s in slices if s is not validation for data in s]
            train = np.array(train)
            validation = np.array(validation)
            yield train, validation