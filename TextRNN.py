
import torch
import numpy as np
from torch import nn
import pickle

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    #index = t.topk(x,k,dim=dim,sorted=True)
    print(index)
    return x.gather(dim, index)


# class TextRNN(nn.Module):
#     def __init__(self, opt):
#         super(TextRNN, self).__init__()
#         self.model_name = 'LSTMText'
#         self.opt = opt
#
#         kernel_size = opt.kernel_size
#         self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
#
#         self.title_lstm = nn.LSTM(input_size=opt.embedding_dim, \
#                                   hidden_size=opt.hidden_size,
#                                   num_layers=opt.num_layers,
#                                   bias=True,
#                                   batch_first=False,
#                                   # dropout = 0.5,
#                                   bidirectional=True
#                                   )
#         self.content_lstm = nn.LSTM(input_size=opt.embedding_dim, \
#                                     hidden_size=opt.hidden_size,
#                                     num_layers=opt.num_layers,
#                                     bias=True,
#                                     batch_first=False,
#                                     # dropout = 0.5,
#                                     bidirectional=True
#                                     )
#         print('opt.embedding_dim,opt.hidden_size,opt.num_layers',opt.embedding_dim,opt.hidden_size,opt.num_layers)
#         print(self.content_lstm)
#         #self.pooling =
#         # self.dropout = nn.Dropout()
#         self.fc = nn.Sequential(
#             nn.Linear(opt.kmax_pooling * (opt.hidden_size * 2 ), opt.linear_hidden_size),
#             nn.BatchNorm1d(opt.linear_hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(opt.linear_hidden_size, opt.num_classes)
#         )
#         # self.fc = nn.Linear(3 * (opt.title_dim+opt.content_dim), opt.num_classes)
#         if opt.embedding_path:
#             print('load embedding')
#             embed_file = pickle.load(open(opt.embedding_path, 'rb'))
#             embedding = np.asarray([line[1:] for line in embed_file])
#
#             self.encoder.weight.data.copy_(t.FloatTensor(embedding))
#
#     def forward(self, content):
#         #title = self.encoder(title)
#         content = self.encoder(content)
#         #print('input content shape',content.shape)
#         #title_out = self.title_lstm(title.permute(1, 0, 2))[0].permute(1, 2, 0)
#         print(type(content))
#         print(len(content))
#         #content_out = self.content_lstm(content.permute(1, 0, 2))[0].permute(1, 2, 0)
#         content_out = self.content_lstm(content)
#         #print('cont_out shape',cont_out.shape)
#         #title_conv_out = kmax_pooling((title_out), 2, self.opt.kmax_pooling)
#         #print('content_out=',content_out)
#         print(type(content_out))
#         print(len(content_out))
#         content_conv_out = kmax_pooling(content_out, 2, self.opt.kmax_pooling)
#         #print('content_conv_out shape',content_conv_out.shape)
#         conv_out = content_conv_out
#         #conv_out = t.cat((title_conv_out, content_conv_out), dim=1)
#         reshaped = conv_out.view(conv_out.size(0), -1)
#         #print('reshape shape',reshaped.shape)
#         logits = self.fc((reshaped))
#         return logits

    # def forward(self, content):
    #     x = content
    #     # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
    #     x = self.encoder(x)  # 经过embedding,x的维度为(batch_size, time_step, input_size=embedding_dim)
    #
    #     # 隐层初始化
    #     # h0维度为(num_layers*direction_num, batch_size, hidden_size)
    #     # c0维度为(num_layers*direction_num, batch_size, hidden_size)
    #     h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
    #         self.layer_num, x.size(0), self.hidden_size)
    #
    #     c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
    #         self.layer_num, x.size(0), self.hidden_size)
    #
    #     # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
    #     # hn,cn表示最后一个状态?维度与h0和c0一样
    #     out, (hn, cn) = self.content_lstm(x, (h0, c0))
    #
    #     # 我们只需要最后一步的输出,即(batch_size, -1, output_size)
    #     out = self.fc(out[:, -1, :])
    #     return out
class TextRNN(nn.Module):
    def __init__(self, opt):
        super(TextRNN, self).__init__()
        self.opt = opt
        args = opt
        embedding_dim = opt.embedding_dim
        label_num = opt.num_classes
        vocab_size = opt.vocab_size
        self.hidden_size = opt.hidden_size
        self.layer_num = opt.num_layers
        self.bidirectional = True

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # if args.static:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
        #     self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        self.lstm = nn.LSTM(embedding_dim,  # x的特征维度,即embedding_dim
                            self.hidden_size,  # 隐藏层单元数
                            self.layer_num,  # 层数
                            batch_first=True,  # 第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
                            bidirectional=self.bidirectional)  # 是否用双向
        self.fc = nn.Linear(self.hidden_size * 2, label_num) if self.bidirectional else nn.Linear(self.hidden_size,
                                                                                                  label_num)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        print(x.device)
        x = x.cuda()
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, time_step, input_size=embedding_dim)

        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
            self.layer_num, x.size(0), self.hidden_size)
        h0 = h0.cuda()
        c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size) if self.bidirectional else torch.zeros(
            self.layer_num, x.size(0), self.hidden_size)
        c0 = c0.cuda()
        # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
        # hn,cn表示最后一个状态?维度与h0和c0一样
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out.cuda()
        hn = hn.cuda()
        cn = cn.cuda()
        # 我们只需要最后一步的输出,即(batch_size, -1, output_size)
        out = self.fc(out[:, -1, :])
        #print('para device',x.device,h0.device,c0.device,out.device)
        return out

