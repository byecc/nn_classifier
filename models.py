import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as autograd
from pack_embedding import LoadEmbedding
import numpy as np

torch.manual_seed(23)


class CNN(nn.Module):
    def __init__(self, hyperparameter):
        super(CNN, self).__init__()
        self.param = hyperparameter

        V = hyperparameter.n_embed
        D = hyperparameter.embed_dim
        Ci = 1
        Co = hyperparameter.kernel_num
        Ks = hyperparameter.kernel_size

        self.n_embed = hyperparameter.n_embed
        self.embed_dim = hyperparameter.embed_dim
        self.n_label = hyperparameter.n_label

        # self.embed = nn.Embedding(V,D)
        self.embed = LoadEmbedding(V, D)
        self.embed.load_pretrained_embedding(hyperparameter.pretrian_file, hyperparameter.model_dict,
                                             embed_pickle=hyperparameter.embed_pickle, binary=False)
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        if hyperparameter.cuda:
            for conv in self.convs1:
                conv.cuda()
        self.dropout = nn.Dropout(hyperparameter.dropout)
        self.linear = nn.Linear(len(Ks) * Co, self.n_label)
        self.batch_size = hyperparameter.batch_size

    def forward(self, x):
        x = self.embed(x)
        # x = F.max_pool1d(x.permute(0, 2, 1), x.size()[1])# simple nn
        # logit = self.linear(x.view(x.size()[0], self.embed_dim)
        x = x.unsqueeze(1)  # CNN model
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.linear(x)
        return logit


class LSTM(nn.Module):
    def __init__(self, hyperparameter):
        super(LSTM, self).__init__()
        V = hyperparameter.n_embed
        D = hyperparameter.embed_dim
        H = hyperparameter.hidden_dim
        L = hyperparameter.n_label

        self.hidden_dim = H
        self.num_layers = hyperparameter.num_layers
        self.batch_size = hyperparameter.batch_size
        self.embedding = LoadEmbedding(V, D)
        if hyperparameter.pretrain:
            self.embedding.load_pretrained_embedding(hyperparameter.pretrain_file, hyperparameter.vocab,
                                                     embed_pickle=hyperparameter.embed_pickle, binary=False)
        self.lstm = nn.LSTM(D, H, num_layers=self.num_layers, batch_first=True)
        if hyperparameter.init_weight:
            init.xavier_normal(self.lstm.all_weights[0][0], gain=hyperparameter.gain)
            init.xavier_normal(self.lstm.all_weights[0][1], gain=hyperparameter.gain)
            # init.xavier_uniform(self.lstm.all_weights[0][1],gain=hyperparameter.gain)
            # init.xavier_uniform(self.lstm.all_weights[0][1],gain=hyperparameter.gain)
        self.hidden2tag = nn.Linear(H, L)
        self.hidden = self.init_hidden(hyperparameter)
        self.dropout = nn.Dropout(hyperparameter.dropout)

    def init_hidden(self, parameter):
        if parameter.cuda:
            return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda(),
                    autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)),
                    autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)))

    def forward(self, sentence, len_x):
        embed = self.embedding(sentence)
        # embed = self.dropout(embed)
        # print(embed)
        # lstm
        embed = nn.utils.rnn.pack_padded_sequence(embed, len_x, batch_first=True)
        lstm_out, self.hidden = self.lstm(embed)
        lstm_out = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = torch.transpose(lstm_out[0], 1, 2)
        # pooling
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # lstm_out = self.dropout(lstm_out)
        # linear
        logit = self.hidden2tag(lstm_out)
        # logit = F.log_softmax(logit,1)
        # logit = self.dropout(logit)
        return logit


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, hyperparameter):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = hyperparameter.embed_dim
        self.hidden_dim = hyperparameter.hidden_dim
        self.out_dim = hyperparameter.n_label
        self.add_cuda = hyperparameter.cuda

        self.ix = nn.Linear(self.in_dim, self.hidden_dim)
        self.ih = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fx = nn.Linear(self.in_dim, self.hidden_dim)
        self.fh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ox = nn.Linear(self.in_dim, self.hidden_dim)
        self.oh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ux = nn.Linear(self.in_dim, self.hidden_dim)
        self.uh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss_func = nn.NLLLoss()

        if hyperparameter.cuda:
            self.loss_func.cuda()

    def node_forward(self, inputs, child_c, child_h):
        child_sum = F.torch.sum(child_h, 0, keepdim=True)

        i = F.sigmoid(self.ix(inputs) + self.ih(child_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_sum))

        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)

        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        # c = F.torch.mul(i,u)+F.torch.sum(fc,0)
        # h = F.torch.mul(o,F.tanh(c))

        c = i * u + F.torch.sum(fc, 1, keepdim=True)
        h = o * F.tanh(c)

        c = torch.sum(c, 0)
        h = torch.sum(h, 0)

        return c, h

    def forward(self, tree, embeds):
        # embeds = self.embedding(embeds)
        loss = autograd.Variable(torch.FloatTensor(1))
        gold = autograd.Variable(torch.LongTensor([tree.label]))
        if self.add_cuda:
            loss = loss.cuda()
            gold = gold.cuda()
        for child in tree.children:
            _, child_loss = self.forward(child, embeds)
            loss = loss + child_loss
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embeds[tree.value - 1], child_c, child_h)

        output = self.out(tree.state[1])
        output = self.softmax(output)
        loss = loss + self.loss_func(output, gold)

        return output, loss

    def get_child_states(self, tree):
        """
        get c and h of all children
        :param tree:
        :return:
        """
        num_children = len(tree.children)
        if num_children == 0:
            child_c = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
            child_h = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
            if self.add_cuda:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = autograd.Variable(torch.zeros(num_children, 1, self.hidden_dim))
            child_h = autograd.Variable(torch.zeros(num_children, 1, self.hidden_dim))
            for idx, child in enumerate(tree.children):
                child_c[idx] = child.state[0]
                child_h[idx] = child.state[1]
            if self.add_cuda:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        return child_c, child_h


class BatchChildSumTreeLSTM(nn.Module):
    def __init__(self, hyperparameter):
        super(BatchChildSumTreeLSTM, self).__init__()
        self.in_dim = hyperparameter.embed_dim
        self.hidden_dim = hyperparameter.hidden_dim
        self.out_dim = hyperparameter.n_label
        self.add_cuda = hyperparameter.cuda
        V = hyperparameter.n_embed
        D = hyperparameter.embed_dim
        self.embedding = LoadEmbedding(V, D)
        if hyperparameter.pretrain:
            self.embedding.load_pretrained_embedding(hyperparameter.pretrain_file, hyperparameter.vocab,
                                                     requires_grad=hyperparameter.fine_tune,
                                                     embed_pickle=hyperparameter.embed_save_pickle, binary=False)

        self.ix = nn.Linear(self.in_dim, self.hidden_dim)
        self.ih = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fx = nn.Linear(self.in_dim, self.hidden_dim)
        self.fh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ox = nn.Linear(self.in_dim, self.hidden_dim)
        self.oh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ux = nn.Linear(self.in_dim, self.hidden_dim)
        self.uh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.softmax = nn.LogSoftmax()
        self.loss_func1 = nn.NLLLoss(reduce=False)
        self.loss_func2 = nn.NLLLoss(reduce=True)


    def node_forward(self, inputs, child_c, child_h):
        child_sum = F.torch.sum(child_h, 0, keepdim=True)

        i = F.sigmoid(self.ix(inputs) + self.ih(child_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_sum))

        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)

        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        # c = F.torch.mul(i,u)+F.torch.sum(fc,0)
        # h = F.torch.mul(o,F.tanh(c))

        c = i * u + F.torch.sum(fc, 1, keepdim=True)
        h = o * F.tanh(c)

        c = torch.sum(c, 0)
        h = torch.sum(h, 0)

        return c, h

    def batch_forward(self,inputs,child_c,child_h,child_h_sum,child_f_sum):
        child_c = autograd.Variable(child_c)
        child_h = autograd.Variable(child_h)
        child_h_sum = autograd.Variable(child_h_sum)
        child_f_sum = autograd.Variable(child_f_sum)
        inputs = torch.unsqueeze(inputs,1)
        i = F.sigmoid(self.ix(inputs)+self.ih(child_h_sum))
        f = F.sigmoid(self.fx(inputs)+self.fh(child_h))
        o = F.sigmoid(self.ox(inputs)+self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs)+self.uh(child_h_sum))

        c = i*u+child_f_sum+F.mul(f,child_c)
        h = o*F.tanh(c)

        return f,c,h

    def forward(self, forest,gold):
        level = forest.max_level
        while level >=0:
            nodes = []
            for node in forest.node_list:
                if node.level == level:
                    nodes.append(node)
            sentence = autograd.Variable(torch.LongTensor([n.word_idx for n in nodes]))
            inputs = self.embedding(sentence)
            nlen = len(nodes)

            # child_c = autograd.Variable(torch.zeros(nlen,1,self.hidden_dim))
            # child_h = autograd.Variable(torch.zeros(nlen,1,self.hidden_dim))
            # child_h_sum = autograd.Variable(torch.zeros(nlen,1,self.hidden_dim))
            # child_f_sum = autograd.Variable(torch.zeros(nlen,1,self.hidden_dim))

            child_c = torch.zeros(nlen, 1, self.hidden_dim)
            child_h = torch.zeros(nlen, 1, self.hidden_dim)
            child_h_sum = torch.zeros(nlen, 1, self.hidden_dim)
            child_f_sum = torch.zeros(nlen, 1, self.hidden_dim)

            for idx,node in enumerate(nodes):
                num = len(node.children)
                if num > 0:
                    h_list = []
                    f = []
                    c =[]
                    for child in node.children:
                        h_list.append(child.state[1])
                        f.append(child.f)
                        c.append(child.state[0])
                    child_h_sum[idx] = F.torch.sum(torch.cat(h_list),0,keepdim=True)
                    child_f_sum[idx] = F.torch.sum(F.mul(torch.cat(f),torch.cat(c)),0,keepdim=True)
                child_c[idx] = node.state[0]
                child_h[idx] = node.state[1]
            f,c,h = self.batch_forward(inputs,child_c,child_h,child_h_sum,child_f_sum)

            out = self.out(h)
            out = torch.squeeze(out,1)

            # if level!=0:
            #     out_loss = self.loss_func1(self.softmax(out),gold)
            #     for idx, node in enumerate(nodes):
            #         node.state = (c[idx], h[idx])
            #         node.f = f[idx]
            #         for child in node.children:
            #             out_loss.data[idx] += child.loss.data[0]
            #         node.loss = out_loss[idx]
            # else:
            #     out_loss =self.loss_func2(self.softmax(out),gold)
            #     return out,out_loss

            if level == 0:
                out = self.softmax(out)
                out_loss = self.loss_func2(out, gold)
                return out,out_loss
            else:
                for idx, node in enumerate(nodes):
                    node.state = (c.data[idx], h.data[idx])
                    node.f = f.data[idx]
            level -=1

class BinaryTreeLeafModule(nn.Module):
    def __init__(self, hyperparameter):
        super(BinaryTreeLeafModule, self).__init__()
        self.in_dim = hyperparameter.embed_dim
        self.hidden_dim = hyperparameter.hidden_dim

        self.cx = nn.Linear(self.in_dim, self.hidden_dim)
        self.ox = nn.Linear(self.in_dim, self.hidden_dim)

    def forward(self, x):
        c = self.cx(x)
        o = F.sigmoid(self.ox(c))
        h = o * F.tanh(c)
        return c, h


class BinaryTreeComposer(nn.Module):
    def __init__(self, hyperparameter):
        super(BinaryTreeComposer, self).__init__()
        self.in_dim = hyperparameter.embed_dim
        self.hidden_dim = hyperparameter.hidden_dim

        def new_gate():
            lh = nn.Linear(self.in_dim, self.hidden_dim)
            rh = nn.Linear(self.in_dim, self.hidden_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()

    def forward(self, lc, lh, rc, rh):
        i = F.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.rflh(lh) + self.lfrh(rh))
        u = F.tanh(self.ulh(lh) + self.urh(rh))
        c = i * u + lf * lc + rf * rc
        h = F.tanh(c)

        return c, h


class BinaryTreeLstm(nn.Module):
    def __init__(self, hyperparameter):
        super(BinaryTreeLstm, self).__init__()
        self.in_dim = hyperparameter.embed_dim
        self.hidden_dim = hyperparameter.hidden_dim
        self.cuda = hyperparameter.cuda
        self.out_dim = hyperparameter.n_label

        self.leaf_module = BinaryTreeLeafModule(hyperparameter)
        self.composer = BinaryTreeComposer(hyperparameter)

        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        self.softmax = nn.LogSoftmax()
        self.nllloss = nn.NLLLoss()

    def forward(self, tree, embeds):
        loss = autograd.Variable(torch.zeros(1))

        num_children = len(tree.children)
        if num_children == 0:
            tree.state = self.leaf_module.forward(embeds[tree.position - 1])
        else:
            for child in tree.children:
                output, child_loss = self.forward(child, embeds)
                loss += child_loss
            lc, lh, rc, rh = self.get_child_state(tree)
            tree.state = self.composer.forward(lc, lh, rc, rh)

        output = self.out(tree.state[1])
        loss += self.nllloss(F.log_softmax(output), autograd.Variable(torch.LongTensor([tree.label])))

        return output, loss

    @staticmethod
    def get_child_state(tree):
        lc, lh = tree.children[0].state
        rc, rh = tree.children[1].state
        return lc, lh, rc, rh


class EmbeddingModel(nn.Module):
    def __init__(self, hyperparameter):
        super(EmbeddingModel, self).__init__()
        V = hyperparameter.n_embed
        D = hyperparameter.embed_dim
        self.embedding = LoadEmbedding(V, D)
        if hyperparameter.pretrain:
            self.embedding.load_pretrained_embedding(hyperparameter.pretrain_file, hyperparameter.vocab,
                                                     requires_grad=hyperparameter.fine_tune,
                                                     embed_pickle=hyperparameter.embed_save_pickle, binary=False)

    def forward(self, sentence):
        return self.embedding(sentence)
