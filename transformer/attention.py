import math

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(4321)
np.random.seed(4321)

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

        score = k @ torch.transpose(q, 0, 1) / math.sqrt(k.shape[-1])
        if mask is not None:
            score = score.masked_fill_(mask == 0, float('-inf'))
        att_array = nn.functional.softmax(score, dim=1)

        att_score = att_array @ v

        return score, att_score

    def forward(self, inputs, mask=None):
        '''
        :param inputs: n * d_s
        :return:
        '''
        k, q, v = self.wk(inputs), self.wq(inputs), self.wv(inputs)
        #print(k.shape)

        attention_score = self.attention(k, q, v, mask)

        return attention_score


inputs = torch.ones((4, 10)).float()

att_function = Attention(inputs.shape[-1])
mask = torch.tril(torch.ones((inputs.shape[0],inputs.shape[0])))
att_score = att_function.forward(inputs, mask)

print(att_score)
print(att_score[0].shape, att_score[1].shape)

print(mask)
