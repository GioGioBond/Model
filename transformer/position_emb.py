import torch
import torch.nn as nn
import time
inputs = torch.randint(0,10, (4,8))
print(inputs)

start_time = time.time()
def postion_embedding(inputs):
    '''
    2i->sin(pos/10000^(2*i/dim))
    2i+1->cos(pos/10000^(2*i/dim))

    :param inputs: n*dim
    :return:
    '''

    pe = torch.zeros(inputs.shape)
    # for _p in range(inputs.shape[0]):
    #
    #     tmp = torch.tensor(_p/pow(10000, torch.arange(0, inputs.shape[1], 2)/inputs.shape[1]))
    #     #print(tmp)
    #     print(torch.arange(0, inputs.shape[1], 2))
    #     print(tmp)
    #     # tmp = torch.tensor(_p/pow(10000, 2*_d/inputs.shape[1]))
    #     pe[_p, 0::2] = torch.sin(tmp)
    #     pe[_p, 1::2] = torch.cos(tmp)
    #     #pass
    tmp = torch.tensor(torch.arange(0, inputs.shape[0]) / pow(10000, torch.arange(0, inputs.shape[1], 2) / inputs.shape[1]))
    pe[:, 0::2] = torch.sin(tmp)
    pe[:, 1::2] = torch.cos(tmp)
    return pe

pe = postion_embedding(inputs)
end_time = time.time()
print(end_time - start_time)
print(pe)
print('-'*60)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  # dropout是原文的0.1，max_len原文没找到
        '''max_len是假设的一个句子最多包含5000个token'''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 开始位置编码部分,先生成一个max_len * d_model 的矩阵，即5000 * 512
        # 5000是一个句子中最多的token数，512是一个token用多长的向量来表示，5000*512这个矩阵用于表示一个句子的信息
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # pos：[max_len,1],即[5000,1]
        print(torch.arange(0, max_len, dtype=torch.float))
        print('pos',pos)
        # 先把括号内的分式求出来,pos是[5000,1],分母是[256],通过广播机制相乘后是[5000,256]
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        print(torch.arange(0, d_model, 2))
        print(torch.arange(0, d_model, 2).float())
        print(pow(10000.0, torch.arange(0, d_model, 2).float() / d_model))
        print('div_term',div_term)
        # 再取正余弦
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        # 一个句子要做一次pe，一个batch中会有多个句子，所以增加一维用来和输入的一个batch的数据相加时做广播
        pe = pe.unsqueeze(0)  # [5000,512] -> [1,5000,512]
        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''x: [batch_size, seq_len, d_model]'''
        # 5000是我们预定义的最大的seq_len，就是说我们把最多的情况pe都算好了，用的时候用多少就取多少
        x = x + self.pe[:, :x.size(1), :]
        print(self.pe[:, :x.size(1), :])
        return self.dropout(x)  # return: [batch_size, seq_len, d_model], 和输入的形状相同

start_time = time.time()
pe = PositionalEncoding(d_model = inputs.shape[1], max_len = inputs.shape[0])(pe)
end_time = time.time()
print(end_time - start_time)

