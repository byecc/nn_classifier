from data_utils import *
import numpy as np

class Forest:

    def __init__(self,trees):

        self.trees = trees
        self.node_list = trees_to_nodes(trees)
        self.adj_mat = nodes_to_adjmat(self.node_list)
        self.max_level = get_max_level(self.node_list)

def trees_to_nodes(trees):
    node_list = []
    for tree in trees:
        tree.forest_ix = len(node_list)
        node_list.append(tree)
        for child in tree.children:
            child.forest_ix = len(node_list)
            node_list.append(child)
            add_forest_ix(child,node_list)
    return node_list

def add_forest_ix(tree,node_list):
    for child in tree.children:
        child.forest_ix = len(node_list)
        node_list.append(child)
        add_forest_ix(child,node_list)

def nodes_to_adjmat(node_list):
    v = len(node_list)
    matrix = np.zeros((v,v))
    for i in range(v):
        for j in range(v):
            if node_list[j] in node_list[i].children:
                matrix[i][j] = 1
    return matrix

def get_max_level(node_list):
    return max([n.level for n in node_list])


