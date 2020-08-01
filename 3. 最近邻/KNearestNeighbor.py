import numpy as np
from collections import Counter

class KNearestNeighbor():
    def __init__(self):
        pass

    # 定义距离度量函数
    def compute_distances(self, X_train, X_test, p):
        num_test = X_test.shape[0]
        num_train = X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # 二范数
        # M = np.dot(X_test, X_train.T)
        # te = np.square(X_test).sum(axis=1)
        # tr = np.square(X_train).sum(axis=1)
        # dists = np.sqrt(-2 * M + tr + np.matrix(te).T)

        for i in range(num_test):
            a = np.expand_dims(X_test[i], 0)
            a = np.tile(a, (num_train, 1))
            dists[i] = np.linalg.norm(a - X_train, ord=p, axis=1)

        return dists


    # 使用多数表决的分类决策规则定义预测函数，这里假设k值取1
    def predict_labels(self, y_train, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            closest_y = []
            labels = y_train[np.argsort(dists[i, :])].flatten()
            closest_y = labels[0:k]

            c = Counter(closest_y)
            y_pred[i] = c.most_common(1)[0][0]

        return y_pred


    # k折交叉验证
    def k_cross_validation(self, X_train, y_train, num_folds, k_choices, p):
        X_train_folds = []
        y_train_folds = []

        X_train_folds = np.array_split(X_train, num_folds)
        y_train_folds = np.array_split(y_train, num_folds)
        k_to_accuracies = {}

        for k in k_choices:
            for fold in range(num_folds):
                # 对传入的训练集单独划出一个验证集作为测试集
                X_valid = X_train_folds[fold]
                y_valid = y_train_folds[fold]
                X_train_temp = np.concatenate(X_train_folds[:fold] + X_train_folds[fold + 1:])
                y_train_temp = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])
                
                # 计算距离
                temp_dists = self.compute_distances(X_train_temp, X_valid, p=p)
                temp_y_pred = self.predict_labels(y_train_temp, temp_dists, k=k)
                # temp_y_pred = temp_y_pred.reshape((-1, 1))

                # 查看分类准确率
                num_correct = np.sum(temp_y_pred == y_valid)
                num_test = X_valid.shape[0]
                accuracy = float(num_correct) / num_test
                k_to_accuracies[k] = k_to_accuracies.get(k, []) + [accuracy]

            
        # 打印不同k值不同折数下的分类准确率
        for k in sorted(k_to_accuracies):    
            for accuracy in k_to_accuracies[k]:
                print('k = %d, accuracy = %f' % (k, accuracy))

        accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
        for i in range(len(k_choices)):
            print('k为{}时，平均精度{}'.format(k_choices[i], accuracies_mean[i]))

        best_k = k_choices[np.argmax(accuracies_mean)]
        print('最佳k值为{}'.format(best_k)) 

        return best_k, k_to_accuracies