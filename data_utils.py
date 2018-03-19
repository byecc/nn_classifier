import re
import random
from nltk.parse import stanford
import time
import sys
import torch
import torch.autograd as autograd

random.seed(5)
class Instance:
    def __init__(self):
        self.sentence = ''
        self.label = '-1'

    def show(self):
        print(self.sentence,' ',self.label)

class Code:
    def __init__(self):
        self.code_list = []
        self.label = []

    def show(self):
        print(self.code_list,' ',self.label)

class Node:
    def __init__(self):
        self.word = ""
        self.word_index = ''
        self.parent_index = ''
        self.label = ''
        self.flag = False
        self.relation = ''

class Tree:
    def __init__(self,value,label,level,word,word_idx):
        self.value = value
        self.children = []
        self.state = (torch.zeros(1,150),torch.zeros(1,150)) #save c and h
        self.label = label
        self.level = level #depth of tree
        self.forest_ix = 0
        self.loss = 0
        self.f = torch.zeros(1,150)
        self.word = word
        self.word_idx = word_idx

    def add_child(self,child_tree):
        self.children.append(child_tree)

class Process:
    def __init__(self,path=None,clean_switch=False):
        self.result = []
        self.load_file(path,clean_switch)

    def load_file(self,path,clean_switch):
        with open(path,'r') as f:
            for line in f:
                if clean_switch:
                    line = DataUtils.clean_str(line)
                info = line.split(' ',1)
                inst = Instance()
                inst.sentence = info[1].split()
                inst.label = info[0]
                self.result.append(inst)

class DataUtils:

    @staticmethod
    def create_voca(result):
        vocabulary = {}
        for r in result:
            for s in r.sentence:
                if s not in vocabulary.keys():
                    vocabulary[s] = len(vocabulary)+1
                else:
                    pass
        vocabulary['-unknown-'] = len(vocabulary)+1
        vocabulary['-padding-'] = 0
        return vocabulary

    @staticmethod
    def cross_validation(path, packet_nums, encoding='UTF-8',clean_switch=False):
        result = []
        packet_list = []
        with open(path,'r',encoding=encoding) as fin:
            for line in fin:
                if clean_switch:
                    line = DataUtils.clean_str(line)
                info = line.split(' ',1)
                inst = Instance()
                # print(line)
                assert len(info) == 2
                inst.sentence = info[1].split()
                inst.label = info[0]
                result.append(inst)
        random.shuffle(result)
        length =  len(result)
        packet_len = length//packet_nums
        if length % packet_nums != 0 :
            packet_len += 1
        packet = []
        for i,r in enumerate(result):
            if i%packet_len == 0  and i!=0:
                packet_list.append(packet)
                packet = [r]
            else:
                packet.append(r)
        if len(packet) > 0:
            packet_list.append(packet)
        return packet_list

    @staticmethod
    def read_data(path,encoding='UTF-8',clean_switch = False):
        result = []
        with open(path,'r',encoding=encoding) as fin:
            for line in fin:
                if clean_switch:
                    line = DataUtils.clean_str(line)
                info = line.split(' ', 1)
                inst = Instance()
                # print(line)
                assert len(info) == 2
                inst.sentence = info[1].split()
                inst.label = info[0]
                result.append(inst)

    @staticmethod
    def extract_sentences(rpath,wpath,encoding='utf8',clean_switch=False):
        with open(rpath,'r',encoding=encoding) as fin,open(wpath,'a') as fout:
            for line in fin:
                line = line.strip()
                if clean_switch:
                    line  = DataUtils.clean_str(line)
                fout.write(line.split(' ',1)[1]+'\n')

    @staticmethod
    def extract_dependency_tree(treepath,savepath):
        with open(treepath,'r') as fin,open(savepath,'a') as fout:
            i=1
            for line in fin:
                if line.startswith("Dependency Parse (enhanced plus plus dependencies):"):
                    line = fin.readline()
                    while line!='\n':
                        fout.write(line)
                        line = fin.readline()
                    fout.write('\n')
                    # fout.flush()
                    print(i)
                    i+=1
                else:
                    pass
            print('..')


    @staticmethod
    def clean_str(string):
        """
                Tokenization/string cleaning for all datasets except for SST.
                Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def encode(result,vocabulary):
        encodes = []
        for r in result:
            code = Code()
            for s in r.sentence:
                if s in vocabulary.keys():
                    code.code_list.append(vocabulary[s])
                else:
                    code.code_list.append(vocabulary['-unknown-'])
            if r.label == '0':
                code.label=0
            elif r.label == '1':
                code.label=1
            else:
                raise RuntimeError("label index out of process ")
            encodes.append(code)
        return encodes

    @staticmethod
    def reverse(string):
        return string[::-1]

    @staticmethod
    def build_tree_from_file(file_name,result,vocab):
        trees = []
        with open(file_name,'r') as fin:
            nodes = []
            i=0
            for line in fin:
                line = line.strip()
                info = line.split(' ',1)
                words = info[1].split(' ')
                for word in words:
                    node = Node()
                    w_info = word.split('_')
                    node.label = int(info[0])
                    if w_info[0].endswith("'''"):
                        node.word_index = int(w_info[0].split("'''")[0])
                    else:
                        node.word_index = int(w_info[0])
                    if w_info[1].endswith("'''"):
                        node.parent_index = int(w_info[1].split("'''")[0])
                    else:
                        node.parent_index = int(w_info[1])
                    node.relation = w_info[2]
                    nodes.append(node)
                sentences = result[i].sentence
                for n in nodes:
                    if n.relation == 'root':
                        if sentences[n.word_index-1] in vocab:
                            word_idx =vocab[sentences[n.word_index-1]]
                        else:
                            word_idx = vocab['-unknown-']
                        tree = Tree(n.word_index,n.label,0,sentences[n.word_index-1],word_idx)
                        child_value = n.word_index
                        # nodes.remove(n)
                        n.flag = True
                        DataUtils.add_tree(tree,child_value,nodes,1,sentences,vocab)
                        trees.append(tree)
                nodes = []
                i+=1
        return trees

    @staticmethod
    def build_tree_conll(file_name,result,vocab):
        trees = []
        i=0
        with open(file_name,'r') as fin:
            nodes = []
            for line in fin:
                if line!='\n':
                    # print(line)
                    sentence = result[i].sentence
                    lines = line.strip().split()
                    node = Node()
                    node.word = lines[1]
                    node.word_index = int(lines[0])
                    node.parent_index = int(lines[5])
                    node.relation = lines[6]
                    node.label = int(result[i].label)
                    nodes.append(node)
                else:
                    for node in nodes:
                        if node.relation == "ROOT":
                            if sentence[node.word_index - 1] in vocab:
                                word_idx = vocab[sentence[node.word_index - 1]]
                            else:
                                word_idx = vocab['-unknown-']
                            tree = Tree(node.word_index, node.label, 0, sentence[node.word_index - 1], word_idx)
                            child_value = node.word_index
                            node.flag = True
                            DataUtils.add_tree(tree, child_value, nodes, 1, sentence, vocab)
                            trees.append(tree)
                    nodes = []
                    i+=1
        return trees

    @staticmethod
    def add_tree(tree,child_value,nodes,level,sentences,vocab):
        for node in nodes:
            if child_value == node.parent_index and node.flag is False:
                if sentences[node.word_index-1] in vocab:
                    word_idx = vocab[sentences[node.word_index-1]]
                else:
                    word_idx = vocab['-unknown-']
                child_tree = Tree(node.word_index,node.label,level,sentences[node.word_index-1],word_idx)
                tree.add_child(child_tree)
                node_index = node.word_index
                # nodes.remove(node)
                node.flag = True
                DataUtils.add_tree(child_tree,node_index,nodes,level+1,sentences,vocab)








