import numpy as np
import pandas as pd
import math

class DesitionTree:
    # 定义节点类 二叉树
    class Node:
        def __init__(self, parent = None, label = None, feature_name = None, feature = None):
            self.parent = parent
            self.label = label
            self.feature_name = feature_name
            self.feature = feature
            self.child = {}

            self.result = {
                'label:': self.label,
                'feature_name': self.feature_name,
                'child' : self.child,
            }

        def __repr__(self):
            return '{}'.format(self.result)

        def add_node(self, val, node):
            self.child[val] = node

        def predict(self, features):
            if self.parent == None:
                return self.label

            return self.child[features[self.feature]].predict(features)


    def __init__(self, type = 'id3', epsilon=0.1):
        self.type = type
        self.tree = None
        self.eta = epsilon


    # 数据集D的经验熵H(D)
    @staticmethod
    def entropy(datasets, class_col = -1):
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
        cond_ent = sum([len(datas) / data_len * DesitionTree.entropy(datas) for c, datas in feature_split_data.items()])
        
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
    def choose_best_feature(datasets, type = 'id3'):
        # 样本维度
        feature_count = len(datasets[0])
        # 经验熵
        ent = DesitionTree.entropy(datasets)
        # 准则
        rules = []

        for feature_col in range(feature_count - 1):
            cond_ent = DesitionTree.cond_entropy(datasets, feature_col)

            if type == 'id3':
                info_gain = ent - cond_ent  # 信息增益
                rules.append((feature_col, info_gain))
            elif type == 'c4.5':
                feature_ent = DesitionTree.cond_entropy(datasets, feature_col)
                info_gain_ratio = (ent - cond_ent) / feature_ent # 信息增益比
                rules.append((feature_col, info_gain_ratio))

        # print(rules)
        # 找到准则最大的特征
        best_feature = max(rules, key = lambda x : x[-1])
        return best_feature


    def train(self, train_data, parent_node = None):
        """
        input:数据集D(DataFrame格式，label位于最后一列)，特征集A，阈值eta
        output:决策树T
        """

        y_train, features = train_data.iloc[:, -1], train_data.columns[:-1]
        
        # 1.若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return DesitionTree.Node(parent = parent_node, label = y_train.iloc[0])

        # 2.若特征集A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            max_label = y_train.value_counts(ascending = False).iloc[0]
            return DesitionTree.Node(parent = parent_node, label = max_label)

        # 3.计算特征集A中各特征对D的信息增益(或信息增益比)，选择最大的特征Ag
        max_feature_col, max_feature_rule_value = DesitionTree.choose_best_feature(np.array(train_data), self.type)
        max_feature_name = features[max_feature_col]

        # 4.如果Ag的信息增益(或信息增益比)小于阈值eta，则置T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，返回T
        if max_feature_rule_value < self.eta:
            max_label = y_train.value_counts(ascending = False).iloc[0]
            return DesitionTree.Node(parent = parent_node, label = max_label)

        # 5.否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子结点，由结点及其子结点构成树T，返回T；
        node = DesitionTree.Node(parent = parent_node, feature_name = max_feature_name, feature = max_feature_col)
        
        max_feature_value_list = train_data[max_feature_name].value_counts().index
        for v in max_feature_value_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == v].drop([max_feature_name], axis = 1)
            sub_node = self.train(sub_train_df)
            node.add_node(v, sub_node)

        return node


    def fit(self, train_data):
        self.tree = self.train(train_data)
        return self.tree


    def predict(self, X_test):
        return self.tree.predict(X_test)


# test
# 书上题目5.1
def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]

    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels


datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)

# max_feature_col, max_feature_rule_value = DesitionTree.choose_best_feature(datasets, 'id3')
# print(max_feature_col)
# print(max_feature_rule_value)

dt = DesitionTree()
tree = dt.fit(data_df)
print(tree)
pred = dt.predict(['老年', '否', '否', '一般'])
print(pred)