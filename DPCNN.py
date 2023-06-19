import torch
import torch.nn as nn
import numpy as np
import pickle

torch.manual_seed(1)


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x


class DPCNN(nn.Module):
    def __init__(self, opt):
        super(DPCNN, self).__init__()
        self.model_name = "DPCNN"
        self.opt = opt

        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(opt.embedding_dim, opt.inception_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=opt.inception_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.region_embedding_t = nn.Sequential(
            nn.Conv1d(opt.embedding_dim, opt.inception_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=opt.inception_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=opt.inception_dim),
            nn.ReLU(),
            nn.Conv1d(opt.inception_dim, opt.inception_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=opt.inception_dim),
            nn.ReLU(),
            nn.Conv1d(opt.inception_dim, opt.inception_dim, kernel_size=3, padding=1),
        )
        self.conv_block_t = nn.Sequential(
            nn.BatchNorm1d(num_features=opt.inception_dim),
            nn.ReLU(),
            nn.Conv1d(opt.inception_dim, opt.inception_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=opt.inception_dim),
            nn.ReLU(),
            nn.Conv1d(opt.inception_dim, opt.inception_dim, kernel_size=3, padding=1),
        )

        self.num_seq = opt.max_seq_len
        # self.title_seq = opt.title_max_len
        # self.num_seq = opt.sent_len + opt.title_len
        resnet_block_list = []
        resnet_block_list_t = []
        while (self.num_seq >= 2):
            resnet_block_list.append(ResnetBlock(opt.inception_dim))
            self.num_seq = self.num_seq // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        # while (self.title_seq >= 2):
        #     resnet_block_list_t.append(ResnetBlock(opt.inception_dim))
        #     self.title_seq = self.title_seq // 2
        self.resnet_layer_t = nn.Sequential(*resnet_block_list_t)
        self.fc = nn.Sequential(
            # nn.Linear(opt.inception_dim * (self.num_seq + self.title_seq), opt.linear_hidden_size),
            nn.Linear(opt.inception_dim * self.num_seq, opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(opt.linear_hidden_size, opt.num_classes)
        )
        # if opt.embedding_path:
        #     print('load embedding')
        #     embed_file = pickle.load(open(opt.embedding_path, 'rb'))
        #     #embedding = np.asarray([line[1:] for line in embed_file])
        #     embedding = np.asarray([embed_file[line] for line in range(len(embed_file))])
        #     self.embedding.weight.data.copy_(torch.FloatTensor(embedding))

    # def forward(self, x):
    def forward(self, x):
        x = self.embedding(x)
        # title = self.embedding(title)
        #print('foward x shape')
        #print(x.shape)
        x = x.permute(0, 2, 1)
        # title = title.permute(0, 2, 1)

        x = self.region_embedding(x)
        # title = self.region_embedding_t(title)

        # x = torch.cat((x, title), 2)
        x = self.conv_block(x)
        # title = self.conv_block_t(title)
        x = self.resnet_layer(x)
        # title = self.resnet_layer_t(title)
        x = x.permute(0, 2, 1)
        # title = title.permute(0, 2, 1)
        x = x.contiguous().view(x.shape[0], -1)
        # title = title.contiguous().view(x.shape[0], -1)
        # input = torch.cat((x, title), dim=1)
        out = self.fc(x)
        return out