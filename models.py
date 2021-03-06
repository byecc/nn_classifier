import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as autograd
from pack_embedding import LoadEmbedding
import numpy as np
import time

torch.manual_seed(233)
# torch.cuda.manual_seed(233)
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
                            requires_grad=hyperparameter.fine_tune,embed_pickle=hyperparameter.embed_save_pickle, binary=False)
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
        self.bid = hyperparameter.bidirectional

        self.ix = nn.Linear(self.in_dim, self.hidden_dim)
        self.ih = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fx = nn.Linear(self.in_dim, self.hidden_dim)
        self.fh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ox = nn.Linear(self.in_dim, self.hidden_dim)
        self.oh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ux = nn.Linear(self.in_dim, self.hidden_dim)
        self.uh = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        self.biout = nn.Linear(self.hidden_dim*2 , self.out_dim)
        self.loss_func = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(hyperparameter.dropout)

        self.lstmcell = nn.LSTMCell(self.in_dim,self.hidden_dim)

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
        child_c = child_h = None
        if self.add_cuda:
            forest_loss = autograd.Variable(torch.zeros(1).cuda())
        else:
            forest_loss = autograd.Variable(torch.zeros(1))
        for level in range(forest.max_level+1)[::-1]:
            nodes = [node for node in forest.node_list if node.level == level]
            nlen = len(nodes)
            input_ix = []
            chi_par = {}
            max_childs,row = 0,0
            offset_pos,fx_offset,hc_offset,fc_offset = [],[],[],[]
            for idx, node in enumerate(nodes):
                input_ix.append(node.forest_ix)
                childs = []
                if len(node.children) > max_childs:
                    max_childs = len(node.children)
                for ch_ix, child in enumerate(node.children):
                    childs.append(ch_ix)
                chi_par[idx] = childs
            if child_h is None:  # if no child nodes
                max_childs = 1
                # offset_pos = [0 for i in range(nlen)]
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
                    child_h = autograd.Variable(torch.zeros(nlen, self.hidden_dim).cuda())
                    child_c = autograd.Variable(torch.zeros(nlen, self.hidden_dim).cuda())
                child_h_sum = autograd.Variable(torch.zeros(nlen * max_childs, self.hidden_dim).cuda())
                child_fh = autograd.Variable(torch.zeros(fx_len, self.hidden_dim).cuda())
                child_fc = autograd.Variable(torch.zeros(fx_len, self.hidden_dim).cuda())
                child_fc_sum = autograd.Variable(torch.zeros(nlen * max_childs, self.hidden_dim).cuda())
                select_indices = autograd.Variable(torch.LongTensor(input_ix).cuda())
                offset_pos = autograd.Variable(torch.LongTensor(offset_pos).cuda())
                fx_offset = autograd.Variable(torch.LongTensor(fx_offset).cuda())
                hc_offset = autograd.Variable(torch.LongTensor(hc_offset).cuda())
                fc_offset = autograd.Variable(torch.LongTensor(fc_offset).cuda())
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

            # embed_input = torch.index_select(embeds, 0, select_indices)
            # test_start = time.time()
            embed_input = embeds[select_indices]

            fh = self.fh(child_h)
            if len(offset_pos) > 0:
                child_h_sum.index_copy_(0, offset_pos, child_h)
                child_fh.index_copy_(0, hc_offset, fh)
                child_fc.index_copy_(0, hc_offset, child_c)
                f = F.sigmoid(self.fx(embed_input[fx_offset]) + child_fh)
                fc = F.mul(f, child_fc)
                child_fc_sum.index_copy_(0, fc_offset, fc)

            child_h_sum = child_h_sum.view([nlen, max_childs, self.hidden_dim])
            child_fc_sum = child_fc_sum.view([nlen, max_childs, self.hidden_dim])


            child_c, child_h = self.batch_forward(embed_input, torch.sum(child_h_sum, 1), torch.sum(child_fc_sum, 1))

            out = self.out(self.dropout(child_h))
            out = torch.unsqueeze(out, 1)
            # test_start = time.time()
            for idx, node in enumerate(nodes):
                node.dt_state = torch.unsqueeze(child_c[idx],0),torch.unsqueeze(child_h[idx],0)
            #     if node.label is not None:
            #         if self.add_cuda:
            #             node_gold = autograd.Variable(torch.LongTensor([node.label]).cuda())
            #         else:
            #             node_gold = autograd.Variable(torch.LongTensor([node.label]))
            #         # out_list.append(out[idx])
            #         # gold_list.append(node_gold)
            #         forest_loss += self.loss_func(out[idx], node_gold)

            # print("test time:",time.time()-test_start)
            if level == 0:
                # forest_loss = self.loss_func(torch.cat(out_list),torch.cat(gold_list))
                # return torch.squeeze(out, 1), forest_loss
                return torch.squeeze(out, 1)
            # level -= 1

    def bid_forward(self,forest,embeds):
        self.forward(forest,embeds)
        c = h = None
        # forest_h = autograd.Variable(torch.zeros(len(forest.trees),forest.max_nodes,self.hidden_dim*2))
        if self.add_cuda:
            forest_h = autograd.Variable(torch.zeros(len(forest.trees)*forest.max_nodes,self.hidden_dim*2).cuda())
        else:
            forest_h = autograd.Variable(torch.zeros(len(forest.trees)*forest.max_nodes,self.hidden_dim*2))
        # forest_h = [[ autograd.Variable(torch.zeros(1,self.hidden_dim*2)).detach() for j in range(forest.max_nodes)]for i in range(len(forest.trees))]
        pos = [0 for i in range(len(forest.trees))]
        for level in range(forest.max_level+1):
            nodes = [node for node in forest.node_list if node.level == level]
            nlen = len(nodes)
            input_ix = []
            c_list,h_list,fc_list,offset_list,cat_h_list = [],[],[],[],[]
            for idx,node in enumerate(nodes):
                input_ix.append(node.forest_ix)
                if node.parent is not None:
                    c_list.append(node.parent.td_state[0])
                    h_list.append(node.parent.td_state[1])
            if len(c_list) == 0:
                if self.add_cuda:
                    c = autograd.Variable(torch.zeros(nlen,self.hidden_dim).cuda())
                    h = autograd.Variable(torch.zeros(nlen,self.hidden_dim).cuda())
                else:
                    c = autograd.Variable(torch.zeros(nlen, self.hidden_dim))
                    h = autograd.Variable(torch.zeros(nlen, self.hidden_dim))
            else:
                c = torch.cat(c_list)
                h = torch.cat(h_list)
            if self.add_cuda:
                input_ix = autograd.Variable(torch.LongTensor(input_ix).cuda())
            else:
                input_ix = autograd.Variable(torch.LongTensor(input_ix))
            embeds_input = embeds[input_ix]
            # embeds_input = torch.unsqueeze(embeds_input,1)
            lstm_time = time.time()
            nodes_c,nodes_h = self.lstmcell(embeds_input,(h,c))
            print("lstm time:",time.time()-lstm_time)
            nodes_h = self.dropout(nodes_h)
            for idx, node in enumerate(nodes):
                # node_c = torch.cat([node.state[0],parent_c[idx]])
                node.td_state = torch.unsqueeze(nodes_c[idx],0),torch.unsqueeze(nodes_h[idx],0)
                node_h = torch.cat([node.dt_state[1],node.td_state[1]],1)
                cat_h_list.append(node_h)
                # forest_h[node.mark][pos[node.mark]] = node_h
                # pos[node.mark] += 1
                offset_list.append(node.mark*forest.max_nodes+pos[node.mark])
                pos[node.mark] += 1
            if self.add_cuda:
                offset = autograd.Variable(torch.LongTensor(offset_list).cuda())
            else:
                offset = autograd.Variable(torch.LongTensor(offset_list))
            forest_h.index_copy_(0,offset,torch.cat(cat_h_list))
        forest_h = forest_h.view([len(forest.trees),forest.max_nodes,self.hidden_dim*2])
        forest_h = forest_h.permute(0,2,1)
        forest_h = F.max_pool1d(forest_h,forest_h.size(2)).squeeze(2)
        out  = self.biout(forest_h)
        return out

    # def forward(self,forest,embeds):
    #     level = forest.max_level
    #     child_c = child_h = None
    #     nodes_len = []
    #     input_ixs, offset_poss, fx_offsets, hc_offsets, fc_offsets ,max_childss,fx_lens = [], [], [], [], [],[],[]
    #     for i in range(level+1):
    #         nodes = [node for node in forest.node_list if node.level == i]
    #         nodes_len.append(len(nodes))
    #         chi_par = {}
    #         max_childs, row = 1, 0
    #         input_ix,offset_pos, fx_offset, hc_offset, fc_offset = [], [], [], [],[]
    #         for idx, node in enumerate(nodes):
    #             input_ix.append(node.forest_ix)
    #             childs = []
    #             if len(node.children) > max_childs:
    #                 max_childs = len(node.children)
    #             for ch_ix, child in enumerate(node.children):
    #                 childs.append(ch_ix)
    #             chi_par[idx] = childs
    #         #level max_childs
    #         max_childss.append(max_childs)
    #         for key, val in chi_par.items():
    #             if len(val) > 0:
    #                 for v in val:
    #                     offset_pos.append(key * max_childs + v)
    #                     fx_offset.append(key)
    #                     hc_offset.append(row)
    #                     row += 1
    #                     fc_offset.append(key * max_childs + v)
    #             else:
    #                 row += 1
    #                 fx_offset.append(key)
    #                 fc_offset.append(key * max_childs)
    #         input_ixs.append(input_ix)
    #         offset_poss.append(offset_pos)
    #         fx_offsets.append(fx_offset)
    #         hc_offsets.append(hc_offset)
    #         fc_offsets.append(fc_offset)
    #         fx_lens.append(len(fx_offset))
    #     input_ixs,input_ixs_len = self.pad_list(input_ixs)
    #     offset_poss,offset_poss_len = self.pad_list(offset_poss)
    #     fx_offsets,fx_offsets_len = self.pad_list(fx_offsets)
    #     hc_offsets,hc_offsets_len = self.pad_list(hc_offsets)
    #     fc_offsets,fc_offsets_len = self.pad_list(fc_offsets)
    #     if self.add_cuda:
    #         input_ixs = torch.LongTensor(input_ixs).cuda()
    #         input_ixs_len = torch.LongTensor(input_ixs_len).cuda()
    #         offset_poss = autograd.Variable(torch.LongTensor(offset_poss).cuda())
    #         offset_poss_len = torch.LongTensor(offset_poss_len).cuda()
    #         fx_offsets = autograd.Variable(torch.LongTensor(fx_offsets).cuda())
    #         fx_offsets_len = torch.LongTensor(fx_offsets_len).cuda()
    #         hc_offsets = autograd.Variable(torch.LongTensor(hc_offsets).cuda())
    #         hc_offsets_len = torch.LongTensor(hc_offsets_len).cuda()
    #         fc_offsets = autograd.Variable(torch.LongTensor(fc_offsets).cuda())
    #         fc_offsets_len = torch.LongTensor(fc_offsets_len).cuda()
    #     else:
    #         input_ixs = torch.LongTensor(input_ixs)
    #         input_ixs_len = torch.LongTensor(input_ixs_len)
    #         offset_poss = autograd.Variable(torch.LongTensor(offset_poss))
    #         offset_poss_len = torch.LongTensor(offset_poss_len)
    #         fx_offsets = autograd.Variable(torch.LongTensor(fx_offsets))
    #         fx_offsets_len = torch.LongTensor(fx_offsets_len)
    #         hc_offsets = autograd.Variable(torch.LongTensor(hc_offsets))
    #         hc_offsets_len = torch.LongTensor(hc_offsets_len)
    #         fc_offsets = autograd.Variable(torch.LongTensor(fc_offsets))
    #         fc_offsets_len = torch.LongTensor(fc_offsets_len)
    #     while level >= 0 :
    #         if self.add_cuda:
    #             if child_h is None:
    #                 child_h = autograd.Variable(torch.zeros(nodes_len[level], self.hidden_dim).cuda())
    #                 child_c = autograd.Variable(torch.zeros(nodes_len[level], self.hidden_dim).cuda())
    #             child_h_sum = autograd.Variable(torch.zeros(nodes_len[level] * max_childss[level], self.hidden_dim).cuda())
    #             child_fh = autograd.Variable(torch.zeros(fx_lens[level], self.hidden_dim).cuda())
    #             child_fc = autograd.Variable(torch.zeros(fx_lens[level], self.hidden_dim).cuda())
    #             child_fc_sum = autograd.Variable(torch.zeros(nodes_len[level] * max_childss[level], self.hidden_dim).cuda())
    #             # select_indices = autograd.Variable(torch.LongTensor(input_ixs[level]).cuda())
    #             # offset_pos = autograd.Variable(torch.LongTensor(offset_poss[level]).cuda())
    #             # fx_offset = autograd.Variable(torch.LongTensor(fx_offsets[level]).cuda())
    #             # hc_offset = autograd.Variable(torch.LongTensor(hc_offsets[level]).cuda())
    #             # fc_offset = autograd.Variable(torch.LongTensor(fc_offsets[level]).cuda())
    #         else:
    #             if child_h is None:
    #                 child_h = autograd.Variable(torch.zeros(nodes_len[level], self.hidden_dim))
    #                 child_c = autograd.Variable(torch.zeros(nodes_len[level], self.hidden_dim))
    #             child_h_sum = autograd.Variable(torch.zeros(nodes_len[level] * max_childss[level], self.hidden_dim))
    #             child_fh = autograd.Variable(torch.zeros(fx_lens[level], self.hidden_dim))
    #             child_fc = autograd.Variable(torch.zeros(fx_lens[level], self.hidden_dim))
    #             child_fc_sum = autograd.Variable(torch.zeros(nodes_len[level] * max_childss[level], self.hidden_dim))
    #             # select_indices = autograd.Variable(torch.LongTensor(input_ixs[level]))
    #             # offset_pos = autograd.Variable(torch.LongTensor(offset_poss[level]))
    #             # fx_offset = autograd.Variable(torch.LongTensor(fx_offsets[level]))
    #             # hc_offset = autograd.Variable(torch.LongTensor(hc_offsets[level]))
    #             # fc_offset = autograd.Variable(torch.LongTensor(fc_offsets[level]))
    #
    #         embed_input = embeds[input_ixs[level][:input_ixs_len[level]]]
    #
    #         fh = self.fh(child_h)
    #         if offset_poss_len[level] > 0:
    #             child_h_sum.index_copy_(0, offset_poss[level][:offset_poss_len[level]], child_h)
    #             child_fh.index_copy_(0, hc_offsets[level][:hc_offsets_len[level]], fh)
    #             child_fc.index_copy_(0, hc_offsets[level][:hc_offsets_len[level]], child_c)
    #             f = F.sigmoid(self.fx(embed_input[fx_offsets[level][:fx_offsets_len[level]]]) + child_fh)
    #             fc = F.mul(f, child_fc)
    #             child_fc_sum.index_copy_(0, fc_offsets[level][:fc_offsets_len[level]], fc)
    #
    #         child_h_sum = child_h_sum.view([nodes_len[level], max_childss[level], self.hidden_dim])
    #         child_fc_sum = child_fc_sum.view([nodes_len[level], max_childss[level], self.hidden_dim])
    #
    #         child_c, child_h = self.batch_forward(embed_input, torch.sum(child_h_sum, 1), torch.sum(child_fc_sum, 1))
    #
    #         out = self.out(self.dropout(child_h))
    #         out = torch.unsqueeze(out, 1)
    #
    #         if level == 0:
    #             # forest_loss = self.loss_func(torch.cat(out_list),torch.cat(gold_list))
    #             # return torch.squeeze(out, 1), forest_loss
    #             return torch.squeeze(out, 1)
    #         level -= 1

    def pad_list(self,o_list):
        l_list = [len(i) for i in o_list]
        max_len = max(l_list)
        n_list = [item+[0]*(max_len-len(item)) for item in o_list]
        return n_list,l_list

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
        else:
            self.embedding.weight = nn.Parameter(torch.randn((V, D)), requires_grad=hyperparameter.fine_tune)

        if hyperparameter.cuda:
            s = self.state_dict()
            self.state_dict()['embedding.weight'].cuda()

    def forward(self, sentence):
        return self.embedding(sentence)
