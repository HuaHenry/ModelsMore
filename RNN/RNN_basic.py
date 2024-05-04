# -*- coding: utf-8 -*-
'''
@File    :   code.py
@Time    :   2024/05/04 23:46:52
@Author  :   Zhouqi Hua 
@Version :   1.0
@Desc    :   None
'''

# 1 - 基础版 RNN

import torch
print(torch.__file__)

from torch import nn

class Rnn(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(Rnn, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(32, 1)     # 每个时间步输出的处理方式

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state
