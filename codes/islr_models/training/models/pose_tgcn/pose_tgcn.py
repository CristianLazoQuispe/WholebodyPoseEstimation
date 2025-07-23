#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
import torch.nn.functional as F



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,num_joints, bias=True, init_A=0):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(num_joints, num_joints))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # AHW
        support = torch.matmul(input, self.weight)  # HW
        output = torch.matmul(self.att, support)  # g
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):

    def __init__(self, in_features, num_joints, p_dropout, bias=True, is_resi=True):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_resi = is_resi

        self.gc1 = GraphConvolution(in_features, in_features,num_joints)
        #self.bn1 = nn.BatchNorm1d(num_joints * in_features)

        self.gc2 = GraphConvolution(in_features, in_features,num_joints)
        #self.bn2 = nn.BatchNorm1d(num_joints * in_features)
        self.bn1 = nn.BatchNorm1d(in_features)
        self.bn2 = nn.BatchNorm1d(in_features)  

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        #y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.bn1(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        #y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.bn2(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.act_f(y)
        y = self.do(y)
        if self.is_resi:
            y = y + x
            return y
        else:
            return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class PoseGTCN(nn.Module):
    def __init__(self, input_feature,num_joints, hidden_feature, num_class, p_dropout, num_stage=1, is_resi=True):
        super(PoseGTCN, self).__init__()
        self.num_stage = num_stage
        self.num_joints = num_joints
        self.gc1 = GraphConvolution(input_feature, hidden_feature,num_joints)
        self.bn1 = nn.BatchNorm1d(num_joints * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature,num_joints, p_dropout=p_dropout, is_resi=is_resi))

        self.gcbs  = nn.ModuleList(self.gcbs)
        self.do    = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x, masks=None):
        batch_size, seq_len, features = x.shape
        num_joints = features // 2
        x = x.view(batch_size, seq_len, num_joints, 2)
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_joints, seq_len, 2]
        x = x.reshape(batch_size, num_joints, seq_len * 2)
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)
        out = torch.mean(y, dim=1)
        out = self.fc_out(out)

        return out


if __name__ == '__main__':
    model = PoseGTCN(input_feature=100, num_joints=135, hidden_feature=32, num_class=100, p_dropout=0.5, num_stage=2, is_resi=False)
