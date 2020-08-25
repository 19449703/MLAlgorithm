import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

# 问题太多了，弄不明白太恶心了，不弄了！！！！！！！！！！！！！！
class DecisionTree:
    class Node:
        def __init__(self, parent = None, label = None, feature_name = None, split_value = None):
            self.parent = parent
            self.label = label
            self.feature_name = feature_name
            self.split_value = split_value  # for gini
            self.child = {}

        def add_node(self, val, node):
            self.child[val] = node

        def predict(self, data):
            if len(self.child) == 0:
                return self.label
            
            key = data.loc[self.feature_name]
            if self.split_value != None:
                # for gini
                key = self.split_value == data.loc[self.feature_name]

            return self.child[key].predict(data)


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
        cond_ent = sum([len(datas) / data_len * DecisionTree.entropy(datas) for c, datas in feature_split_data.items()])
        
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
        feature_count_radio = [feature_data.count(value) / data_len for value in feature_set]
        
        ent = -sum([ratio * math.log(ratio, 2) for ratio in feature_count_radio])
        # ent=0时取什么值合适？？？
        return ent or 0.01


    # 样本集合D的基尼指数
    @staticmethod
    def gini(datasets, class_col = -1):
        # 样本数量
        data_len = len(datasets)
        # 类别列
        class_col_data = [data[class_col] for data in datasets]
        # 类别集合
        class_set = set(class_col_data)
        # 每个类别的样本数量
        class_count = {c : class_col_data.count(c) for c in class_set}
        # 每个类别的概率
        class_probs = {c : class_count[c] / data_len for c in class_set}

        gini_value = sum([prob * (1 - prob) for c, prob in class_probs.items()]) 
        return gini_value


    # 给定特征A的条件下，集合D的基尼指数
    @staticmethod
    def cond_Gini(datasets, feature_col, class_col = -1):
        # 样本数量
        data_len = len(datasets)
        # 类别集合
        value_set = set([data[feature_col] for data in datasets])

        # 计算针对特征A每一个可能的值的基尼指数
        gini_value = {}
        for feature_value in value_set:
            # 特征A的一列数据
            feature_data = [data[feature_col] for data in datasets]
            # 根据特征A是否取某一可能值a，集合D分割成D1和D2两部分
            d1 = [data for data in datasets if data[feature_col] == feature_value]
            d2 = [data for data in datasets if data[feature_col] != feature_value]
            # D1和D2的基尼指数
            gini_d1 = DecisionTree.gini(d1, class_col)
            gini_d2 = DecisionTree.gini(d2, class_col)

            gini_value[feature_value] = len(d1) / data_len * gini_d1 + len(d2) / data_len * gini_d2

        # 找到最小的基尼指数及对应的切分点
        split = min(gini_value, key = lambda x : x)
        gini_min = gini_value[split]
        return gini_min, split 


    # 选择信息增益最大的特征
    @staticmethod
    def choose_best_feature_on_info_gain(datasets):
        # 样本维度
        feature_count = len(datasets[0])
        # 经验熵
        ent = DecisionTree.entropy(datasets)
        # 所有特征的信息增益
        info_gains = []

        for feature_col in range(feature_count - 1):
            cond_ent = DecisionTree.cond_entropy(datasets, feature_col)
            info_gain = ent - cond_ent
            info_gains.append((feature_col, info_gain))

        best_feature = max(info_gains, key = lambda x : x[1])
        return best_feature


    # 选择信息增益比最大的特征
    @staticmethod
    def choose_best_feature_on_info_gain_ratio(datasets):
        # 样本维度
        feature_count = len(datasets[0])
        # 经验熵
        ent = DecisionTree.entropy(datasets)
        # 准则
        info_gain_ratios = []

        for feature_col in range(feature_count - 1):
            cond_ent = DecisionTree.cond_entropy(datasets, feature_col)
            feature_ent = DecisionTree.feature_entropy(datasets, feature_col)
            info_gain_ratio = (ent - cond_ent) / feature_ent # 信息增益比
            info_gain_ratios.append((feature_col, info_gain_ratio))

        best_feature = max(info_gain_ratios, key = lambda x : x[1])
        return best_feature


    # 选择基尼指数最小的特征
    @staticmethod
    def choose_best_feature_on_gini(datasets):
        # 样本维度
        feature_count = len(datasets[0])
        # 准则
        ginis = []

        for feature_col in range(feature_count - 1):
            gini_min, split = DecisionTree.cond_Gini(datasets, feature_col)
            ginis.append((feature_col, gini_min, split))

        best_feature = max(ginis, key = lambda x : x[1])
        return best_feature


    def train(self, train_data, parent_node = None):
        """
        input:数据集D(DataFrame格式，label位于最后一列)，特征集A
        output:决策树T
        """

        y_train, features = train_data.iloc[:, -1], train_data.columns[:-1]
        
        # 1.若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return DecisionTree.Node(parent = parent_node, label = y_train.iloc[0])

        # 2.若特征集A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            max_label = y_train.value_counts(ascending = False).index[0]
            return DecisionTree.Node(parent = parent_node, label = max_label)

        # 3.计算特征集A中各特征对D的 信息增益/信息增益比/基尼指数，选择最大的特征Ag
        if self.type == 'id3':
            best_feature_col, best_feature_rule_value = DecisionTree.choose_best_feature_on_info_gain(np.array(train_data))
        elif self.type == 'c4.5':
            best_feature_col, best_feature_rule_value = DecisionTree.choose_best_feature_on_info_gain_ratio(np.array(train_data))
        elif self.type == 'gini':
            best_feature_col, best_feature_rule_value, split_value = DecisionTree.choose_best_feature_on_gini(np.array(train_data))
        
        max_feature_name = features[best_feature_col]

        # 4.如果Ag的 信息增益/信息增益比/基尼指数 小于阈值eta，则置T为单节点树，并将D中实例数最大的类Ck作为该节点的类标记，返回T
        if best_feature_rule_value < self.eta:
            max_label = y_train.value_counts(ascending = False).index[0]
            return DecisionTree.Node(parent = parent_node, label = max_label)

        # 5.否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的类作为标记，构建子结点，由结点及其子结点构成树T，返回T；
        if self.type == 'id3' or self.type == 'c4.5':
            node = DecisionTree.Node(parent = parent_node, feature_name = max_feature_name)

            max_feature_value_list = train_data[max_feature_name].value_counts().index
            for v in max_feature_value_list:
                sub_train_df = train_data.loc[train_data[max_feature_name] == v].drop([max_feature_name], axis = 1)
                sub_node = self.train(sub_train_df, node)
                node.add_node(v, sub_node)
        elif self.type == 'gini':
            node = DecisionTree.Node(parent = parent_node, feature_name = max_feature_name, split_value = split_value)

            d1 = train_data.loc[train_data[max_feature_name] == split_value].drop([max_feature_name], axis = 1)
            sub_node1 = self.train(d1, node)
            node.add_node(True, sub_node1)

            d2 = train_data.loc[train_data[max_feature_name] != split_value].drop([max_feature_name], axis = 1)
            sub_node2 = self.train(d2, node)
            node.add_node(False, sub_node2)

        return node


    def fit(self, train_data):
        """
        input:数据集D(DataFrame格式，label位于最后一列)
        output:决策树T
        """
        self.tree = self.train(train_data)
        return self.tree


    def predict(self, test_data):
        """
        input:测试数据集(DataFrame格式，label位于最后一列)
        output:预测结果
        """

        x_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
        result = []

        for index, row in x_test.iterrows():
            result.append(self.tree.predict(row))

        return result


    def score(self, test_data):
        y = test_data.iloc[:, -1].tolist()
        pred = self.predict(test_data)
        correct_num = sum([a[0] == a[1] for a in zip(y, pred)])
        return correct_num / len(y)


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


# datasets, labels = create_data()
# train_df = pd.DataFrame(datasets, columns = labels)
# predict_df = pd.DataFrame([['老年', '否', '否', '一般', '?']], columns=labels)

# print(os.path.exists('4. 决策树/Data/example_data.csv'))
# data = pd.read_csv('4. 决策树/Data/example_data.csv', dtype={'windy': 'str'})
# train_df, test_df = train_test_split(data, test_size=0.1)

# print("---------------train_df's label")
# print(train_df.iloc[:, -1].tolist())

# print('---------------id3')
# dt = DecisionTree(type = 'id3')
# tree = dt.fit(train_df)
# print(dt.predict(train_df))
# print(dt.score(train_df))

# print('---------------c4.5')
# dt = DecisionTree(type = 'c4.5')
# tree = dt.fit(train_df)
# print(dt.predict(train_df))
# print(dt.score(train_df))

# print('---------------gini')
# dt = DecisionTree(type = 'gini')
# tree = dt.fit(train_df)
# print(dt.predict(test_df))
# print(dt.score(train_df))

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# iris = load_iris()
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df['label'] = iris.target
# df.columns = [
#     'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
# ]

# df_train, df_test = train_test_split(df, test_size=0.35, random_state=9)

# data = np.array(df)
# X, y = data[:, :-1], data[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=9)

# # scikit-learn实例
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz

# clf = DecisionTreeClassifier('entropy')
# print(clf.fit(X_train, y_train))
# print(clf.predict(X_test))
# print(clf.score(X_test, y_test))

# print('---------------- 漂亮的分割线 ----------------')

# dt = DecisionTree()
# tree = dt.fit(df_train)
# print(dt.predict(df_test))
# print(dt.score(df_test))