# -*- coding: utf-8 -*-
'''
@File    :   RNN_Seq2Seq.py
@Time    :   2024/05/13 16:11:14
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   None
'''

# 2 - RNN进阶: Seq2Seq

import torch
import torch.nn as nn
import torch.optim as optim

# 定义Encoder模型
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)     # view(1, 1, -1) 为了匹配 RNN 的输入
        output = embedded                                   # output 为了和 Decoder 的输入匹配
        output, hidden = self.rnn(output, hidden) 
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# 定义Decoder模型
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)       # view(1, 1, -1) 为了匹配 RNN 的输入
        output = torch.relu(output)                         # relu 作为激活函数
        output, hidden = self.rnn(output, hidden)           # output 为了和 Encoder 的输出匹配
        output = self.softmax(self.out(output[0]))          # output 为了和目标匹配
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        target_length = target.size(0)                      # 目标长度
        encoder_hidden = self.encoder.init_hidden()         # 初始化 Encoder 隐藏层

        input_length = input.size(0)                        # 输入长度
        for ei in range(input_length):                      # Encoder 阶段
            _, encoder_hidden = self.encoder(input[ei], encoder_hidden) # Encoder 隐藏层

        decoder_input = torch.tensor([[START_TOKEN]])       # 开始标记
        decoder_hidden = encoder_hidden                     # Decoder 隐藏层初始化

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                decoder_input = target[di]                  # 使用教师强制作为下一个输入
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()     # 使用预测作为下一个输入

        return decoder_output