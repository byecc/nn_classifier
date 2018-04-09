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
from vocab import Vocab
from Dataset import *

random.seed(133)

class CNNClassifier:
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter

    def train(self, train_set, dev_set=None, test_set=None):
        batch_block = len(train_set) // self.hyperparameter.batch_size
        if len(train_set) % self.hyperparameter.batch_size:
            batch_block += 1
        model = CNN(self.hyperparameter)
        if self.hyperparameter.cuda:
            model.cuda()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
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
                logit = model.forward(x)
                loss = F.cross_entropy(logit, y)
                loss_sum += loss
                loss.backward()
                optimizer.step()
                max_list = self.get_max_index(logit.data)
                for j in range(len(max_list)):
                    if y.data[j] == max_list[j]:
                        cor_sentence += 1
                    sum_sentences += 1
            print("train_set accuracy:{}\n consume time:{}".format(cor_sentence / sum_sentences, time.time() - start))
            if dev_set is not None:
                self.eval(model, dev_set, "dev", self.hyperparameter.cuda)
            if test_set is not None:
                test_accuracy = self.eval(model, test_set, "test", self.hyperparameter.cuda)
                if test_accuracy > max_accuracy:
                    max_accuracy = test_accuracy
        print('time:', time.time() - start)
        return max_accuracy

    def eval(self, model, dataset, dataset_name, cuda):
        model.eval()
        cor = s = 0
        for d in dataset:
            x, y = self.toVariable(d, cuda)
            logit = model(x)
            if y.data[0] == self.get_max_index(logit.data)[0]:
                cor += 1
            s += 1
        print("{} dataset accuracy ：{}".format(dataset_name, cor / s))

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
        parameters = filter(lambda p: p.requires_grad, model.parameters())
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
            print("train_set accuracy:{}\n consume time:{}".format(cor_sentence / sum_sentences, time.time() - start))
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
            self.hyperparameter.embed_pickle = "cv_subj_embed/embed_" + str(i) + ".pkl"  # dim 300 word-vector
            # self.hyperparameter.embed_pickle = "cv_cr_embed/embed_" + str(i) + ".pkl"  #dim 300 word-vector
            # self.hyperparameter.embed_pickle = "cv_mr_embed/embed_" + str(i) + ".pkl"  # dim 300 word-vector
            test_accuracy = self.train(train_set, test_set=test_set)
            sum_acc += test_accuracy
            print("第{}个包：准确率：{}".format(i, test_accuracy))
        print("average accuracy：{}".format(sum_acc / len(packet_list)))

    def eval(self, model, dataset, dataset_name, cuda):
        model.eval()
        cor = s = 0
        for d in dataset:
            x, y = self.toVariable(d, cuda)
            logit = model(x, [x.data.shape[1]])
            if y.data[0] == self.get_max_index(logit.data)[0]:
                cor += 1
            s += 1
        print("{} dataset accuracy ：{}".format(dataset_name, cor / s))

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
        self.args = hyperparameter

    def train(self, train_set, dev_set=None, test_set=None):
        data_len =  len(train_set[0])
        train_trees = train_set[0]
        train_sentences = train_set[1]
        rand = []
        for i in range(data_len):
            rand.append(i)
        batch_block = data_len// self.args.batch_size
        if data_len % self.args.batch_size:
            batch_block += 1

        embed_model = EmbeddingModel(self.args)
        model = BatchChildSumTreeLSTM(self.args)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if self.args.optim == "adam":
            optimizer = optim.Adam(params=parameters, lr=self.args.learn_rate, weight_decay=self.args.weight_decay)
        elif self.args.optim == "adagrad":
            optimizer = optim.Adagrad(params=parameters,lr=self.args.learn_rate,weight_decay=self.args.weight_decay)

        dev_forest = Forest(dev_set[0])
        test_forest = Forest(test_set[0])
        for i in range(self.args.epoch):
            model.train()
            embed_model.train()
            model.zero_grad()
            embed_model.zero_grad()
            loss_sum = num_sentences = cor_sentences =  0
            embed_time = model_time = backward_time = 0
            start = time.time()
            random.shuffle(rand)
            print("第{}次训练:".format(i + 1))
            #no batch
            # optimizer.zero_grad()
            # embed_model.zero_grad()
            # for idx in range(data_len):
            #     if self.args.cuda:
            #         sen = autograd.Variable(torch.LongTensor(train_sentences[rand[idx]])).cuda()
            #     else:
            #         sen = autograd.Variable(torch.LongTensor(train_sentences[rand[idx]]))
            #     embeds = embed_model.forward(sen)
            #     embeds = torch.unsqueeze(embeds,1)
            #     output,loss = model.forward(train_trees[rand[idx]],embeds)
            #     # loss = loss
            #     loss_sum += loss.data[0]
            #     loss.backward()
            #     if idx % self.args.batch_size==0:
            #         for f in embed_model.parameters():
            #             f.data.sub_(f.grad.data*self.args.embed_learn_rate)
            #         optimizer.step()
            #         embed_model.zero_grad()
            #         optimizer.zero_grad()
            #
            #     output[:, 1] = -9999
            #     val,pred = torch.max(output.data,1)
            #     if pred[0] == train_trees[rand[idx]].label:
            #         cor_sentences += 1
            #     num_sentences += 1

            # add batch
            for idx in range(batch_block):
                left = idx * self.args.batch_size
                right = left + self.args.batch_size
                if right < data_len:
                    forest = Forest([train_trees[m] for m in rand[left:right]])
                else:
                    forest = Forest([train_trees[m] for m in rand[left:]])
                y = torch.LongTensor([t.label for t in forest.trees])
                sen = autograd.Variable(torch.LongTensor([n.word_idx for n in forest.node_list]))
                embeds = embed_model.forward(sen)
                embed_time += time.time()-start
                out,loss = model.forward(forest,torch.unsqueeze(embeds,1))
                model_time += time.time() - start
                # loss = loss/self.args.batch_size
                # loss = F.cross_entropy(out,autograd.Variable(y))
                loss_sum += loss.data[0]
                loss.backward()
                backward_time += time.time() - start
                for f in embed_model.parameters():
                    f.data.sub_(f.grad.data*self.args.embed_learn_rate)
                optimizer.step()
                embed_model.zero_grad()
                optimizer.zero_grad()
                # self.args.learn_rate = self.args.learn_rate*0.9
                # optimizer.__setattr__('lr',self.args.learn_rate)
                # model.zero_grad()
                out[:,1] = -9999
                val,pred = torch.max(out.data,1)
                for x in range(len(forest.trees)):
                    if y[x] == pred[x]:
                        cor_sentences += 1
                    num_sentences += 1
                forest.clean_state()

            # print("embed_time:{},model_time:{},backward_time:{}".format(embed_time,model_time,backward_time))
            print("this epoch loss :{},train_accuracy : {},consume time:{}".format(loss_sum / data_len,
                                                                                   cor_sentences / num_sentences,
                                                                                   time.time() - start))
            #no batch
            # if dev_set is not None:
            #     self.test(embed_model,model, dev_set,"dev")
            # if test_set is not None:
            #     self.test(embed_model,model, test_set,"test")
            #batch

            if dev_set is not None:
                self.batchtest(embed_model,model, dev_forest,"dev")
            if test_set is not None:
                self.batchtest(embed_model,model, test_forest,"test")

    def test(self, embed_model,model,dataset, dataset_name):
        model.eval()
        embed_model.eval()
        cor = total = 0
        #no batch
        data_len = len(dataset[0])
        data_trees = dataset[0]
        data_sentences = dataset[1]
        for idx in range(data_len):
            sen = autograd.Variable(torch.LongTensor(data_sentences[idx]),volatile=True)

            embeds = torch.unsqueeze(embed_model(sen), 1)
            output, loss = model(data_trees[idx], embeds)

            output[:, 1] = -9999
            val, pred = torch.max(output.data, 1)
            if pred[0] == data_trees[idx].label:
                cor += 1
            total += 1
        # batch
        # forest = dataset
        # y = torch.LongTensor([t.label for t in forest.trees])
        # out, loss = model(forest)
        # out[:,1] = -9999
        # val,pred = torch.max(out.data,1)
        # for x in range(len(forest.trees)):
        #     if y[x] == pred[x]:
        #         cor += 1
        #     total += 1
        print(dataset_name, total)
        print("{}_set accuracy:{}".format(dataset_name, cor / total))

    def batchtest(self,embed_model,model,dataset,dataset_name):
        model.eval()
        embed_model.eval()
        cor = total = 0
        forest = dataset
        y = torch.LongTensor([t.label for t in forest.trees])
        sen = autograd.Variable(torch.LongTensor([n.word_idx for n in forest.node_list]),requires_grad = False)
        embeds = torch.unsqueeze(embed_model(sen),1)
        out,loss = model(forest,embeds)
        out[:,1] = -9999
        val,pred = torch.max(out.data,1)
        for x in range(len(forest.trees)):
            if y[x] == pred[x]:
                cor += 1
            total += 1
        forest.clean_state()
        print(dataset_name, total)
        print("{}_set accuracy:{}".format(dataset_name, cor / total))

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

    vocab = Vocab("sst/vocab-cased.txt")
    sent_dict = DataUtils.get_sentiment_dict("sst/sentiment_labels.txt","sst/dictionary.txt")
    param.vocab = vocab.labelToIdx
    param.n_embed = len(vocab.labelToIdx)
    param.n_label = 3
    param.embed_dim = 300
    train_dataset = DataUtils.build_deptree("sst/train/sents.txt","sst/train/dlabels.txt","sst/train/dparents.txt",vocab,sent_dict)
    dev_dataset = DataUtils.build_deptree("sst/dev/sents.txt","sst/dev/dlabels.txt","sst/dev/dparents.txt",vocab,sent_dict)
    test_dataset = DataUtils.build_deptree("sst/test/sents.txt","sst/test/dlabels.txt","sst/test/dparents.txt",vocab,sent_dict)
    classifier = TreeLstmClassifier(param)
    classifier.train(train_set=train_dataset,dev_set=dev_dataset,test_set=test_dataset)
    # classifier.train(train_set=dev_dataset,dev_set=dev_dataset,test_set=dev_dataset)

    # cnnclassifier = CNNClassifier(param)
    # cnnclassifier.train(train_set=train_set,dev_set=dev_set,test_set=test_set)
