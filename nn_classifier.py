import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as autograd
import torch.optim as optim
import argparse
from models import *
from data_utils import *
from torch.utils.data import DataLoader, Dataset
import numpy as np
from hyperparameter import *
import time
from tree_batch import Forest

class LSTMClassifier:
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter

    def train(self, train_set, dev_set=None, test_set=None):
        batch_block = len(train_set) // self.hyperparameter.batch_size
        if len(train_set) % self.hyperparameter.batch_size:
            batch_block += 1
        model = LSTM(self.hyperparameter)
        if self.hyperparameter.cuda:
            model.cuda()
        parameters = filter(lambda p:p.requires_grad,model.parameters())
        optimizer = optim.Adam(params=parameters, lr=self.hyperparameter.learn_rate)
        max_accuracy = 0.0
        for i in range(self.hyperparameter.epoch):
            print("第{}次训练....  ".format(i + 1), end='')
            start = time.time()
            random.shuffle(train_set)
            loss_sum = cor_sentence = sum_sentences = 0
            for block in range(batch_block):
                optimizer.zero_grad()
                left = block * self.hyperparameter.batch_size
                right = left + self.hyperparameter.batch_size
                if right <= len(train_set):
                    x, y, len_x = BatchGenerator.create(train_set[left:right], cuda=self.hyperparameter.cuda)
                else:
                    x, y, len_x = BatchGenerator.create(train_set[left:], cuda=self.hyperparameter.cuda)
                # BatchGenerator.show_vocab(x,self.hyperparameter.vocab)
                logit = model.forward(x, len_x)
                loss = F.cross_entropy(logit, y)
                loss_sum += loss
                loss.backward()
                optimizer.step()
                max_list = self.get_max_index(logit.data)
                for j in range(len(max_list)):
                    if y.data[j] == max_list[j]:
                        cor_sentence += 1
                    sum_sentences += 1
            # print("train_set accuracy:{}\n consume time:{}".format(cor/sum,time.time()-start))
            if dev_set is not None:
                self.eval(model, dev_set, "dev", self.hyperparameter.cuda)
            if test_set is not None:
                test_accuracy = self.eval(model, test_set, "test", self.hyperparameter.cuda)
                if test_accuracy > max_accuracy:
                    max_accuracy = test_accuracy
        print('time:', time.time() - start)
        return max_accuracy

    def cv_train(self, packet_list):
        sum_acc = 0.0
        for i, p in enumerate(packet_list):
            train_list = []
            test_list = []
            for j in range(len(packet_list)):
                if i != j:
                    train_list.extend(packet_list[j])
                else:
                    test_list.extend(packet_list[j])
            vocab = DataUtils.create_voca(train_list)
            train_set = DataUtils.encode(train_list, vocab)
            test_set = DataUtils.encode(test_list, vocab)
            self.hyperparameter.vocab = vocab
            self.hyperparameter.n_embed = len(vocab)
            # self.hyperparameter.embed_pickle = "data/embed_" + str(i) + ".pkl"    # dim 100 word-vector
            self.hyperparameter.embed_pickle = "cv_subj_embed/embed_" + str(i) + ".pkl"  #dim 300 word-vector
            # self.hyperparameter.embed_pickle = "cv_cr_embed/embed_" + str(i) + ".pkl"  #dim 300 word-vector
            # self.hyperparameter.embed_pickle = "cv_mr_embed/embed_" + str(i) + ".pkl"  # dim 300 word-vector
            test_accuracy = self.train(train_set, test_set=test_set)
            sum_acc += test_accuracy
            print("第{}个包：准确率：{}".format(i, test_accuracy))
        print("average accuracy：{}".format(sum_acc / len(packet_list)))

    def eval(self, model, dataset, dataset_name, cuda):
        cor = s = 0
        for d in dataset:
            x, y = self.toVariable(d, cuda)
            logit = model(x, [x.data.shape[1]])
            if y.data[0] == self.get_max_index(logit.data)[0]:
                cor += 1
            s += 1
        # print("{} dataset accuracy ：{}".format(dataset_name,cor/s))
        return cor / s

    @staticmethod
    def get_max_index(output):
        max_list = []
        row = output.size()[0]
        col = output.size()[1]
        for i in range(row):
            max_index, max_num = 0, output[i][0]
            for j in range(col):
                tmp = output[i][j]
                if max_num < tmp:
                    max_num = tmp
                    max_index = j
            max_list.append(max_index)
        return max_list

    @staticmethod
    def toVariable(data, cuda):
        if cuda:
            x = autograd.Variable(torch.LongTensor([data.code_list])).cuda()
            y = autograd.Variable(torch.LongTensor([data.label])).cuda()
        else:
            x = autograd.Variable(torch.LongTensor([data.code_list]))
            y = autograd.Variable(torch.LongTensor([data.label]))
        return x, y


class TreeLstmClassifier:
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter

    def train(self, train_set,train_tree,dev_set=None, test_set=None):
        batch_block = len(train_set)//self.hyperparameter.batch_size
        if len(train_set)%self.hyperparameter.batch_size:
            batch_block+=1
        model = BatchChildSumTreeLSTM(self.hyperparameter)
        if self.hyperparameter.cuda:
            model.cuda()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params=parameters,lr = self.hyperparameter.learn_rate)
        model.train()
        for i in range(self.hyperparameter.epoch):
            loss_sum = num_sentences = cor_sentences =0
            start = time.time()
            #add batch
            for idx in range(batch_block):
                optimizer.zero_grad()
                left = idx*self.hyperparameter.batch_size
                right = left+self.hyperparameter.batch_size
                if right<len(train_set):
                    forest = Forest(train_tree[left:right])
                    data = train_set[left:right]
                else:
                    forest = Forest(train_tree[left:])
                    data = train_set[left:]

                y = autograd.Variable(torch.LongTensor([l.label for l in data]))
                out,loss = model.forward(forest,y)
                # loss = F.cross_entropy(out,y)
                loss.backward()
                optimizer.step()
                max_list = self.get_max_index(out.data)
                for x in range(len(max_list)):
                    if y.data[x] == max_list[x]:
                        cor_sentences+=1
                    num_sentences+=1
                # if idx%100 == 0 and idx!=0:
                #     print("loss : {}".format(loss.data[0]))
            print("this epoch loss :{},train_accuracy : {},consume time:{}".format(loss.data[0],
                                                                                   cor_sentences / num_sentences,
                                                                                   time.time() - start))
            self.eval(model,dev_set[0],dev_set[1],"dev")
            self.eval(model,test_set[0],test_set[1],"test")

    def eval(self,model,dataset,tree_dataset,dataset_name):
        model.eval()
        cor = total = 0
        forest = Forest(tree_dataset)
        y = autograd.Variable(torch.LongTensor([l.label for l in dataset]))
        out,loss = model(forest,y)
        max_list = self.get_max_index(out.data)
        for x in range(len(max_list)):
            if y.data[x] == max_list[x]:
                cor+=1
            total+=1
        print("{} accuracy:{}".format(dataset_name,cor/total))

    @staticmethod
    def get_max_index(output):
        max_list = []
        row = output.size()[0]
        col = output.size()[1]
        for i in range(row):
            max_index, max_num = 0, output[i][0]
            for j in range(col):
                tmp = output[i][j]
                if max_num < tmp:
                    max_num = tmp
                    max_index = j
            max_list.append(max_index)
        return max_list

    @staticmethod
    def toVariable(data, cuda):
        if cuda:
            x = autograd.Variable(torch.LongTensor([data.code_list])).cuda()
            y = autograd.Variable(torch.LongTensor([data.label])).cuda()
        else:
            x = autograd.Variable(torch.LongTensor([data.code_list]))
            y = autograd.Variable(torch.LongTensor([data.label]))
        return x, y

class BatchGenerator:

    @staticmethod
    def create(sentence_data, cuda=False):
        max_sentence_size = 0
        num = len(sentence_data)
        for s in sentence_data:
            if max_sentence_size < len(s.code_list):
                max_sentence_size = len(s.code_list)

        sentence_data.sort(key=lambda s: len(s.code_list), reverse=True)
        len_x = [len(s.code_list) for s in sentence_data]
        batch_input = torch.LongTensor(num, max_sentence_size)
        batch_label = torch.LongTensor(num)

        for i in range(num):
            sd = sentence_data[i]
            for j in range(max_sentence_size):
                if j < len(sd.code_list):
                    batch_input[i][j] = sd.code_list[j]
                else:
                    batch_input[i][j] = 0
            batch_label[i] = sd.label

        if cuda:
            batch_input = autograd.Variable(batch_input).cuda()
            batch_label = autograd.Variable(batch_label).cuda()
        else:
            batch_input = autograd.Variable(batch_input)
            batch_label = autograd.Variable(batch_label)

        return batch_input, batch_label, len_x

    @staticmethod
    def show_vocab(batch_input, vocabulary):
        vk = list(vocabulary.keys())
        vv = list(vocabulary.values())
        for i in range(batch_input.data.size()[0]):
            for j in range(batch_input.data.size()[1]):
                if batch_input.data[i][j] != 0:
                    print(vk[vv.index(batch_input.data[i][j])], ' ', batch_input.data[i][j], end='|')
                else:
                    print('padding', end=' 0|')
            print()


if __name__ == "__main__":

    # lstm sentiment classifier use cv
    # param = HyperParameter()
    # # packet_list = DataUtils.cross_validation('data/rt-polarity.all', param.packet_nums, encoding='iso-8859-1',
    # #                                          clean_switch=True)
    # packet_list = DataUtils.cross_validation('dataset/subj.all', param.packet_nums, encoding='iso-8859-1',
    #                                          clean_switch=True)
    # param.embed_dim = 300
    # param.n_label = 2
    # classifier = LSTMClassifier(param)
    # classifier.cv_train(packet_list)

    # treelstm sentiment classifier on SST
    param = HyperParameter()
    p_train = Process('dataset/stsa.binary.train',False)
    p_dev = Process('dataset/stsa.binary.dev',False)
    p_test = Process('dataset/stsa.binary.test',False)
    vocab = DataUtils.create_voca(p_train.result)
    train_set = DataUtils.encode(p_train.result,vocab)
    dev_set = DataUtils.encode(p_dev.result,vocab)
    test_set = DataUtils.encode(p_test.result,vocab)
    train_dependency_tree = DataUtils.build_tree_conll('data/stsa.binary.train.sen.output',p_train.result,vocab)
    dev_dependency_tree = DataUtils.build_tree_conll('data/stsa.binary.dev.sen.output',p_dev.result,vocab)
    test_dependency_tree = DataUtils.build_tree_conll('data/stsa.binary.test.sen.output',p_test.result,vocab)
    param.n_embed = len(vocab)
    param.embed_dim = 300
    param.n_label = 2
    param.vocab = vocab
    classifier = TreeLstmClassifier(param)
    classifier.train(train_set,train_dependency_tree,dev_set=(dev_set,dev_dependency_tree),test_set=(test_set,test_dependency_tree))