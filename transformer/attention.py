import math

import numpy as np
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim_size):
        super(Attention, self).__init__()
        self.wk = torch.nn.Linear(dim_size, dim_size)
        self.wq = torch.nn.Linear(dim_size, dim_size)
        self.wv = torch.nn.Linear(dim_size, dim_size)

    def attention(self, k, q, v, mask=None):
        '''
        h_d : hidden_dim, n: nums of a sequence
        :param k: n*hidden_d
        :param q: n*h_d
        :param v: n*h_d
        :param mask: n*n
        :return:
        softmax(kq/sqrt(dk))v
        '''
        print(k.shape, q.shape, v.shape)

        #score = torch.mul(k, q)
        #score = torch.matmul(k, q)
        score = k @ torch.transpose(q, 0, 1) / math.sqrt(k.shape[-1])
        if mask:
            score.masked_fill_(mask, -1e9)

        att_array = nn.functional.softmax(score, dim=1)

        print(att_array.shape)
        att_score = att_array @ v
        #score = softmax(score)
        return score, att_score

    def forward(self, inputs):
        '''
        :param inputs: n * d_s
        :return:
        '''
        k, q, v = self.wk(inputs), self.wq(inputs), self.wv(inputs)
        print(k.shape)
        attention_score = self.attention(k, q, v)

        return attention_score

# k = torch.randint(0,10, (4,10)).float()
# q = torch.randint(0,10, (4,10)).float()
# v = torch.randint(0,10, (4,10)).float()

inputs = torch.randint(0, 10, (4, 10)).float()

att_function = Attention(inputs.shape[-1])
att_score = att_function.forward(inputs)

print(att_score)
print(att_score[0].shape, att_score[1].shape)