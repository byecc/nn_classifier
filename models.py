import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as autograd
from pack_embedding import LoadEmbedding
import numpy as np

torch.manual_seed(233)


class CNN(nn.Module):
    def __init__(self, hyperparameter):
        super(CNN, self).__init__()
        self.param = hyperparameter

        V = hyperparameter.n_embed
        D = hyperparameter.embed_dim
        C = hyperparameter.n_label
        Ci = 1
        Co = hyperparameter.kernel_num
        Ks = hyperparameter.kernel_size

        self.n_embed = hyperparameter.n_embed
        self.embed_dim = hyperparameter.embed_dim

        # self.embed = nn.Embedding(V,D)
        self.embed = LoadEmbedding(V, D)
        if hyperparameter.pretrain:
            self.embed.load_pretrained_embedding(hyperparameter.pretrain_file, hyperparameter.vocab,
                                                 embed_pickle=hyperparameter.embed_save_pickle, binary=False)
        else:
            self.embed.weight = nn.Parameter(torch.randn((V, D)), requires_grad=hyperparameter.fine_tune)
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        if hyperparameter.cuda:
            for conv in self.convs1:
                conv.cuda()
        self.dropout = nn.Dropout(hyperparameter.dropout)
        self.linear = nn.Linear(len(Ks) * Co, C)
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
        out = self.linear(x)
        return out


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
        self.embedding.weight = nn.Parameter(torch.randn((V, D)), requires_grad=hyperparameter.fine_tune)
        if hyperparameter.pretrain:
            self.embedding.load_pretrained_embedding(hyperparameter.pretrain_file, hyperparameter.vocab,
                                                     embed_pickle=hyperparameter.embed_save_pickle, binary=False)
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
        self.hyperparameter = hyperparameter

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
        self.loss_func = nn.NLLLoss()
        self.dropout = nn.Dropout(hyperparameter.dropout)

        if hyperparameter.cuda:
            self.loss_func.cuda()

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)

        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))

        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)

        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, embeds):
        # embeds = self.embedding(embeds)
        if self.add_cuda:
            loss = autograd.Variable(torch.zeros(1)).cuda()
        else:
            loss = autograd.Variable(torch.zeros(1))
        for child in tree.children:
            _, child_loss = self.forward(child, embeds)
            loss = loss + child_loss
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embeds[tree.value - 1], child_c, child_h)

        output = self.out(self.dropout(tree.state[1]))
        output = self.softmax(output)
        if tree.label is not None:

            if self.add_cuda:
                gold = autograd.Variable(torch.LongTensor([tree.label])).cuda()
            else:
                gold = autograd.Variable(torch.LongTensor([tree.label]))
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
            child_c = autograd.Variable(torch.Tensor(num_children, 1, self.hidden_dim))
            child_h = autograd.Variable(torch.Tensor(num_children, 1, self.hidden_dim))
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

        self.ix = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.ih = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.fx = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.fh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.ox = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.oh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.ux = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.uh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        self.loss_func = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(hyperparameter.dropout)

        if hyperparameter.clip_max_norm is not None:
            nn.utils.clip_grad_norm(self.parameters(), max_norm=hyperparameter.clip_max_norm)

        for p in self.out.parameters():
            nn.init.normal(p.data, 0, 0.01)


    def batch_forward(self, inputs, child_h_sum, child_fc_sum):
        child_h_sum = torch.squeeze(child_h_sum, 1)
        child_fc_sum = torch.squeeze(child_fc_sum, 1)
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))

        c = F.mul(i, u) + child_fc_sum
        h = F.mul(o, F.tanh(c))

        return c, h

    def forward(self, forest,embeds):
        # embeds = self.dropout(embeds)
        level = forest.max_level
        child_c = child_h = None
        forest_loss = autograd.Variable(torch.zeros(1))
        while level >= 0:
            nodes = []
            for node in forest.node_list:
                if node.level == level:
                    nodes.append(node)
            # inputs = []
            nlen = len(nodes)
            input_ix = []
            chi_par = {}
            max_childs = 0
            for idx, node in enumerate(nodes):
                input_ix.append(node.forest_ix)
                childs = []
                if len(node.children) > max_childs:
                    max_childs = len(node.children)
                for ch_ix, child in enumerate(node.children):
                    childs.append(ch_ix)
                chi_par[idx] = childs
                # num = len(node.children)
                # fx = self.fx(embeds[node.forest_ix])
                # if num > 0:
                #     h_sum_list = []
                #     fc_sum_list = []
                #     for child in node.children:
                #         h_sum_list.append(child.state[1])
                #         f = fx+self.fh(child.state[1])
                #         f = F.sigmoid(f)
                #         fc = F.mul(f,child.state[0])
                #         fc_sum_list.append(fc)
                #     h_sum.append(torch.unsqueeze(torch.sum(torch.cat(h_sum_list),0),0))
                #     fc_sum.append(torch.unsqueeze(torch.sum(torch.cat(fc_sum_list),0),0))
                # elif num == 0:
                #     if self.add_cuda:
                #         h_sum.append(autograd.Variable(torch.zeros(1,self.hidden_dim)).cuda())
                #         fc_sum.append(autograd.Variable(torch.zeros(1,self.hidden_dim)).cuda())
                #     else:
                #         h_sum.append(autograd.Variable(torch.zeros(1, self.hidden_dim)))
                #         fc_sum.append(autograd.Variable(torch.zeros(1, self.hidden_dim)))
            offset_pos = []
            fx_offset = []
            hc_offset = []
            fc_offset = []
            if child_h is None:
                max_childs = nlen
            row = 0
            for key, val in chi_par.items():
                if len(val) > 0:
                    for v in val:
                        offset_pos.append(key * max_childs + v)
                        fx_offset.append(key)
                        hc_offset.append(row)
                        row += 1
                        fc_offset.append(key * max_childs + v)
                else:
                    row += 1
                    fx_offset.append(key)
                    fc_offset.append(key * max_childs)
            fx_len = len(fx_offset)
            # node_num = len(input_ix)
            if self.add_cuda:
                if child_h is None:
                    child_h = autograd.Variable(torch.zeros(nlen, self.hidden_dim)).cuda()
                    child_c = autograd.Variable(torch.zeros(nlen, self.hidden_dim)).cuda()
                child_h_sum = autograd.Variable(torch.zeros(nlen * max_childs, self.hidden_dim)).cuda()
                child_fh = autograd.Variable(torch.zeros(fx_len, self.hidden_dim)).cuda()
                child_fc = autograd.Variable(torch.zeros(fx_len, self.hidden_dim)).cuda()
                child_fc_sum = autograd.Variable(torch.zeros(nlen * max_childs, self.hidden_dim)).cuda()
                select_indices = autograd.Variable(torch.LongTensor(input_ix)).cuda()
                offset_pos = autograd.Variable(torch.LongTensor(offset_pos)).cuda()
                fx_offset = autograd.Variable(torch.LongTensor(fx_offset)).cuda()
                hc_offset = autograd.Variable(torch.LongTensor(hc_offset)).cuda()
                fc_offset = autograd.Variable(torch.LongTensor(fc_offset)).cuda()
            else:
                if child_h is None:
                    child_h = autograd.Variable(torch.zeros(nlen, self.hidden_dim))
                    child_c = autograd.Variable(torch.zeros(nlen, self.hidden_dim))
                child_h_sum = autograd.Variable(torch.zeros(nlen * max_childs, self.hidden_dim))
                child_fh = autograd.Variable(torch.zeros(fx_len, self.hidden_dim))
                child_fc = autograd.Variable(torch.zeros(fx_len, self.hidden_dim))
                child_fc_sum = autograd.Variable(torch.zeros(nlen * max_childs, self.hidden_dim))
                offset_pos = autograd.Variable(torch.LongTensor(offset_pos))
                select_indices = autograd.Variable(torch.LongTensor(input_ix))
                fx_offset = autograd.Variable(torch.LongTensor(fx_offset))
                hc_offset = autograd.Variable(torch.LongTensor(hc_offset))
                fc_offset = autograd.Variable(torch.LongTensor(fc_offset))

            embed_input = torch.index_select(embeds, 0, select_indices)
            # embed_input = self.embedding(autograd.Variable(torch.LongTensor([n.word_idx for n in nodes])))
            # fx = self.fx(embed_input)
            fh = self.fh(child_h)
            if len(offset_pos) > 0:
                child_h_sum = child_h_sum.index_copy(0, offset_pos, child_h.detach())
                fx_list = []
                for fo in range(fx_len):
                    fx_list.append(embed_input[fx_offset[fo]])
                child_fh = child_fh.index_copy(0, hc_offset, fh)
                child_fc = child_fc.index_copy(0, hc_offset, child_c)
                f = F.sigmoid(self.fx(torch.cat(fx_list)) + child_fh)
                fc = F.mul(f, child_fc)
                child_fc_sum = child_fc_sum.index_copy(0, fc_offset, fc)

            child_h_sum = child_h_sum.view([nlen, max_childs, self.hidden_dim])
            child_fc_sum = child_fc_sum.view([nlen, max_childs, self.hidden_dim])

            child_c, child_h = self.batch_forward(embed_input, torch.sum(child_h_sum, 1), torch.sum(child_fc_sum, 1))

            out = self.out(self.dropout(child_h))
            out = torch.unsqueeze(out, 1)
            # c = torch.unsqueeze(child_c,1)
            # h = torch.unsqueeze(child_h,1)
            for idx, node in enumerate(nodes):
                # node.state= (c[idx],h[idx])
                # node.out = out[idx]
                if node.label is not None:
                    if self.add_cuda:
                        node_gold = autograd.Variable(torch.LongTensor([node.label])).cuda()
                    else:
                        node_gold = autograd.Variable(torch.LongTensor([node.label]))
                    forest_loss += self.loss_func(out[idx], node_gold)
                    # for child in node.children:
                    #     if child.label is not None:
                    #         forest_loss+=child.loss
                    # node.loss = node_loss
            if level == 0:
                # loss = self.loss_func(out,autograd.Variable(torch.LongTensor([node.label for node in nodes])))
                # return torch.squeeze(out,1),torch.sum(torch.cat([node.loss for node in nodes]))
                return torch.squeeze(out, 1), forest_loss
                # return out,loss
            level -= 1


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
            # emb = torch.load(hyperparameter.embed_save_pickle)
            # self.state_dict()['embedding.weight'].copy_(emb)
            # if hyperparameter.fine_tune is False:
            #     for param in self.parameters():
            #         param.requires_grad = False
            self.embedding.load_pretrained_embedding(hyperparameter.pretrain_file, hyperparameter.vocab,
                                                     requires_grad=hyperparameter.fine_tune,
                                                     embed_pickle=hyperparameter.embed_save_pickle, binary=False)
        else:
            self.embedding.weight = nn.Parameter(torch.randn((V, D)), requires_grad=hyperparameter.fine_tune)

        if hyperparameter.cuda:
            s = self.state_dict()
            self.state_dict()['embedding.weight'].cuda()

    def forward(self, sentence):
        return self.embedding(sentence)
