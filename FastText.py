import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
import pickle

class FastText(nn.Module):
    def __init__(self, opt):
        super(FastText, self).__init__()
        #self.model_name = 'FastText3'
        self.opt = opt

        self.pre2 = nn.Sequential(
            nn.Linear(opt.embedding_dim, opt.embedding_dim),
            nn.BatchNorm1d(opt.embedding_dim),
            nn.ReLU(True)
        )

        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        # print('encode shape')
        # print(opt.vocab_size)  #13264
        # print(opt.embedding_dim)  #200
        # print(opt.linear_hidden_size) #1000
        self.fc = nn.Sequential(
            nn.Linear(opt.embedding_dim , opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size, opt.num_classes)
        )
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


        # if opt.embedding_path:
        #     print('load embedding')
        #     embed_file = pickle.load(open(opt.embedding_path, 'rb'))
        #     #print(len(embed_file))
        #     print(embed_file[0])
        #     #embedding = np.asarray([line[1:] for line in embed_file])
        #     embedding = np.asarray([embed_file[line] for line in range(len(embed_file))])
        #     print('len embedding')
        #     print(embedding[0])
        #     self.encoder.weight.data.copy_(torch.FloatTensor(embedding))

    def forward(self, content):
        # title_em = self.encoder(title)
        #print('content shape')
        #print(content.shape)
        content_em = self.encoder(content)
        #print(content_em.shape)  #158*500*200
        # title_size = title_em.size()
        content_size = content_em.size()

        # title_2 = self.pre1(title_em.contiguous().view(-1, 200)).view(title_size[0], title_size[1], -1)
        content_2 = self.pre2(content_em.contiguous().view(-1, 200)).view(content_size[0], content_size[1], -1)
        #print('content_2 shape')
        # title_ = t.mean(title_2, dim=1)
        #content_ = content_2.view(content_size[0],-1)
        #128 500 200
        content_ = t.mean(content_2, dim=1)
        #print('content.shape:{},content_em.shape:{},content_size:{},content_2.shape:{},content_.shape:{}'.format(content.shape,content_em.shape,content_size,
        #                                                                                             content_2.shape,content_.shape))
        # inputs = t.cat((title_,content_),1)
        # content.shape: torch.Size([64, 2, 500]), content_em.shape: torch.Size(
        #     [64, 2, 500, 200]), content_size: torch.Size([64, 2, 500, 200]), content_2.shape: torch.Size(
        #     [64, 2, 100000]), content_.shape: torch.Size([64, 100000])
        out = self.fc(content_)

        return out










