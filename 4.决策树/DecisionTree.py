import numpy as np

class DesitionTree:
    def __init__(self, type='id3'):
        self.type = type
        self.root = {}

    # 数据集D的经验熵H(D)
    @staticmethod
    def entropy(datasets, class_col=-1):
        # 样本数量
        data_len = len(datasets)
        # 类别列
        class_col_data = [data[class_col] for data in datasets]
        # 类别集合
        class_set = set(class_col_data)
        # 每个类别的样本数量
        class_count = {c : class_col_data.count(c) for c in class_set}
        # 每个类别的率
        class_probs = {c : class_count[c] / data_len for c in class_set}
        # 经验熵
        ent = -sum([prob * math.log(prob, 2) for c, prob in class_probs.items()])

        return ent


    # 特征A对数据集D的经验条件熵H(D|A)
    @staticmethod
    def cond_entropy(datasets, feature_col):
        # 样本数量
        data_len = len(datasets)
        # 特征A的类别集合
        feature_set = set([data[feature_col] for data in datasets])
        # 特征A的每个类别对应的样本集
        feature_split_data = {c : [data for data in datasets if data[feature_col] == c] for c in feature_set}
        # 经验条件熵
        cond_ent = sum([len(datas) / data_len * entropy(datas) for c, datas in feature_split_data.items()])
        
        return cond_ent


    # 训练数据集D关于特征A的值的熵HA(D)
    @staticmethod
    def feature_entropy(datasets, feature_col):
        # 样本数量
        data_len = len(datasets)
        # 特征A的一列数据
        feature_data = [data[feature_col] for data in datasets]
        # 特征A的类别集合
        feature_set = set(feature_data)
        # 特征A的每个类别的样本数占总全样本数之比
        feature_count_radio = [feature_data.Count(value) / data_len for value in feature_set]
        
        ent = -sum([ratio * math.log(ratio, 2) for ratio in feature_count_radio])
        return ent


    # 选择最佳特征
    @staticmethod
    def choose_best_feature(datasets, class_col=-1):
        # 样本维度
        feature_count = len(datasets[0])
        # 经验熵
        ent = entropy(datasets)
        # 准则
        rules = []

        class_col = class_col == -1 and feature_count - 1 or class_col

        for feature_col in range(feature_count):
            # skip class column
            if feature_col == class_col:
                continue

            cond_ent = cond_entropy(datasets, feature_col)

            if self.type == 'id3':
                info_gain = ent - cond_ent  # 信息增益
                rules.append((feature_col, info_gain))
            elif self.type == 'c4.5':
                feature_ent = cond_entropy(datasets, feature_col)
                info_gain_ratio = (ent - cond_ent) / feature_ent # 信息增益比
                rules.append((feature_col, info_gain_ratio))

        # 找到准则最大的特征
        best_feature = max(rules, key = lambda x : x[-1])
        return best_feature


    def train(self, train_data, eta):
        """
        input:数据集D(DataFrame格式)，特征集A阈值eta
        output:决策树T
        """