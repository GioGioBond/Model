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

