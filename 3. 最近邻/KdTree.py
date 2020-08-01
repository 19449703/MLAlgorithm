import math
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class KdNode(object):
    def __init__(self, dom_elt, split):
        self.dom_elt = dom_elt      # k维向量节点(k维空间中的一个样本点)
        self.split = split          # 整数（进行分割维度的序号）

        self.visited = False        # 在搜索时是否被访问过
        self.parent = None          # 父结点
        self.left = None            # 该结点分割超平面左子空间构成的kd-tree
        self.right = None           # 该结点分割超平面右子空间构成的kd-tree
        self.dist = float("inf")    # 该结点距离目标点的欧拉距离


class KdTree(object):
    def __init__(self, data, start_k=0):
        k = len(data[0]) # 数据维度
        self.data = data
        
        # 按第split维划分数据集exset创建KdNode
        def CreateNode(split, data_set, parent_node):
            if not data_set: 
                return None

            # 按要进行分割的维度对数据排序
            data_set.sort(key=lambda x: x[split])
            # 中位数分割点
            split_pos = len(data_set) // 2
            # 中位数位置的数据
            median = data_set[split_pos]
            # 下次分割的维度
            split_next = (split + 1) % k

            node = KdNode(median, split)
            node.parent = parent_node
            node.left = CreateNode(split_next, data_set[:split_pos], node)
            node.right = CreateNode(split_next, data_set[split_pos + 1 :], node)
            
            return node

        # 从第0维分量开始构建kd树,返回根节点
        self.root = CreateNode(start_k, data, None)


    # KDTree的前序遍历
    def preorder(self, node=None):
        node = node or self.root
        print(node.dom_elt)
        if node.left:
            self.preorder(node.left)
        if node.right:
            self.preorder(node.right)


    # 最近邻搜索，递归写的好复杂让人看不明白！！
    def search(self, x):
        # 数据维度
        k = len(x)

        # 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
        result = namedtuple("Result_tuple", "nearest_point  nearest_dist  nodes_visited")

        def travel(kd_node, target, max_dist):
            if kd_node is None:
                return result([0] * k, float("inf"), 0)

            nodes_visited = 1

            # 进行分割的维度
            s = kd_node.split
            # 进行分割的“轴”
            pivot = kd_node.dom_elt

            # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            if target[s] <= pivot[s]:
                # 下一个访问节点为左子树根节点
                nearer_node = kd_node.left
                # 同时记录下右子树
                further_node = kd_node.right
            else:  # 目标离右子树更近
                # 下一个访问节点为右子树根节点
                nearer_node = kd_node.right
                # 同时记录下左子树
                further_node = kd_node.left

            # 进行遍历找到包含目标点的区域
            temp1 = travel(nearer_node, target, max_dist)
            # 以此叶结点作为“当前最近点”
            nearest = temp1.nearest_point
            # 更新最近距离  
            dist = temp1.nearest_dist  

            nodes_visited += temp1.nodes_visited

            if dist < max_dist:
                # 最近点将在以目标点为球心，max_dist为半径的超球体内
                max_dist = dist  

            # 第s维上目标点与分割超平面的距离
            temp_dist = abs(pivot[s] - target[s])
            # 判断超球体是否与超平面相交
            if max_dist < temp_dist:
                # 不相交则可以直接返回，不用继续判断
                return result(nearest, dist, nodes_visited)
            
            # print("包含目标点的叶结点是 ", temp1)

            #----------------------------------------------------------------------
            # 计算目标点与分割点的欧氏距离
            temp_dist = math.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pivot, target)))

            # 如果“更近”
            if temp_dist < dist:  
                nearest = pivot  # 更新最近点
                dist = temp_dist  # 更新最近距离
                max_dist = dist  # 更新超球体半径

            # 检查另一个子结点对应的区域是否有更近的点
            temp2 = travel(further_node, target, max_dist)

            nodes_visited += temp2.nodes_visited

            # 如果另一个子结点内存在更近距离
            if temp2.nearest_dist < dist:  
                nearest = temp2.nearest_point  # 更新最近点
                dist = temp2.nearest_dist  # 更新最近距离

            return result(nearest, dist, nodes_visited)

        # 从根节点开始递归
        return travel(self.root, x, float("inf"))  


    # k近邻搜索，自己实现一个
    def search_k_nearest(self, target, k):
        self._clear_node_visited_state(self.root)
        cur_nearest = self._find_leaf_node_contain_target(self.root, target)

        # 存放距离最近的k个结点
        k_list = []
        self._find_nearest(cur_nearest, target, k_list, k)

        return k_list


    def _clear_node_visited_state(self, node):
        if node == None:
            return

        node.visited = False
        node.dist = float("inf")

        self._clear_node_visited_state(node.left)
        self._clear_node_visited_state(node.right)


    # 查找包含目标点的叶结点
    def _find_leaf_node_contain_target(self, node, target):
        if node.left and target[node.split] <= node.dom_elt[node.split]:
            return self._find_leaf_node_contain_target(node.left, target)
        elif node.right and target[node.split] > node.dom_elt[node.split]:
            return self._find_leaf_node_contain_target(node.right, target)

        return node


    def _find_nearest(self, node, target, k_list, k):
        if node == None:
            return

        if node.visited:
            self._find_nearest(node.parent, target, k_list, k)
            return

        node.visited = True

        # 当前结点到目标点的距离
        dist = math.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(target, node.dom_elt)))
        node.dist = dist

        # 加入到k_list
        if len(k_list) < k:
            k_list.append(node)
            k_list.sort(key = lambda x:x.dist)
        else:
            # 若比列表里最大的小，则去掉最大的，加入这个
            if dist < k_list[-1].dist:
                k_list[-1] = node
                k_list.sort(key = lambda x:x.dist)

        if node.parent == None:
            return

        if len(k_list) == k and abs(node.parent.dom_elt[node.parent.split] - target[node.parent.split]) > k_list[-1].dist:
            # 切分线另一边不会有更近的点，继续往上回撤
            self._find_nearest(node.parent, target, k_list, k)
        else:
            # 切分线另一边可能有更近的点，在另一边找到叶结点
            another_node = node.parent.right if node.parent.left == node else node.parent.left
            if another_node == None:
                self._find_nearest(node.parent, target, k_list, k)
                return

            leaf = self._find_leaf_node_contain_target(another_node, target)
            self._find_nearest(leaf, target, k_list, k)


    def PlotKdTree(self, target, xmin, xmax, ymin, ymax, plotCoord=True, plot=True):
        if len(self.data[0]) != 2:
            pirnt("PlotKdTree2D只能绘制二维TdTree")
            return

        x = [d[0] for d in self.data]
        y = [d[1] for d in self.data]

        plt.rcParams['figure.figsize'] = ((xmax - xmin) * 0.6, (ymax - ymin) * 0.6)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xticks(np.arange(0, xmax + 1, 1))
        plt.yticks(np.arange(0, ymax + 1, 1))
        plt.scatter(x, y, color='black')
        plt.scatter(target[0], target[1], color='red', marker='^', s=62)

        plt.annotate("target", target, xytext=(5, 5), textcoords='offset pixels', color='red', fontsize=12)

        def PlotPartition(node, xmin, xmax, ymin, ymax):
            if node == None:
                return
            
            if plotCoord:
                plt.annotate("({0:.2f},{1:.2f})".format(node.dom_elt[0], node.dom_elt[1]), node.dom_elt, xytext=(5, 5), textcoords='offset pixels', color='black', fontsize=12)

            if node.split == 0:
                plt.plot((node.dom_elt[0], node.dom_elt[0]), (ymin, ymax), color='black', linewidth=1)
                PlotPartition(node.left, xmin, node.dom_elt[0], ymin, ymax)
                PlotPartition(node.right, node.dom_elt[0], xmax, ymin, ymax)
            elif node.split == 1:
                plt.plot((xmin, xmax), (node.dom_elt[1], node.dom_elt[1]), color='black', linewidth=1)
                PlotPartition(node.left, xmin, xmax, ymin, node.dom_elt[1])
                PlotPartition(node.right, xmin, xmax, node.dom_elt[1], ymax)
            

        PlotPartition(self.root, xmin, xmax, ymin, ymax)

        if plot:
            plt.show()


    def PlotKdTreeWithKNearest(self, target, k_list, xmin, xmax, ymin, ymax, plotCoord=True, plot=True):
        self.PlotKdTree(target, xmin, xmax, ymin, ymax, plotCoord, False)
        
        for i in range(len(k_list)):
            plt.scatter(k_list[i].dom_elt[0], k_list[i].dom_elt[1], color='black', marker='o', s=62)
            plt.annotate(str(i + 1), k_list[i].dom_elt, xytext=(15, -25), textcoords='offset pixels', color='blue', fontsize=12, arrowprops=dict(color='blue', arrowstyle="-"))

        if plot:
            plt.show()


# data = [(2,5), (6,1.4), (3.8,8), (8.5,3), (8,5.5), (1,1), (1.5,8.5)]
# targets = [(8, 4.5), (3.3, 7.3), (2, 1.5)]

# kd = KdTree(data, 1)
# k_list = kd.search_k_nearest(targets[0], 1)
# for node in k_list:
#     print(node.dom_elt)