# RNN 循环神经网络 1（原始模型）

> **Author**：Zhouqi Hua, Tongji University
>
> **Email**：henryhua0721@foxmail.com
>
> **Date**：2024/5/4



:star2: **一句话模型总结**

为了更好利用**序列信息**的前后文关系，引入时间序列上实时更新的**隐藏状态**，在下一个时间步作为输入的一部分被传递，从而实现信息的传递。

![image-20240504220325878](http://henry-typora.oss-cn-beijing.aliyuncs.com/img/image-20240504220325878.png)



:star2: **模型架构解释**

使用传统的 CNN 难以处理长序列的问题，而 RNN 通过其**隐藏状态**的传递可以实现前后文的**信息传递**，从而可以处理**序列变化的数据**（如某个单词的含义会根据上文内容而变化）。

具体实现如下：（此处忽略网络偏置）

![image-20240504220719100](http://henry-typora.oss-cn-beijing.aliyuncs.com/img/image-20240504220719100.png)

- $x$：当前时间步的**输入**
- $h$：上一个时间步传递给当前的**历史状态**
- $h^{'}$：根据 $x$ 和 $h$ 获得的传递给下一个时间步的**当前状态**
- $y$：根据 $x$ 和 $h$ 获得的**输出**

通常情况下：

- $h^{'}=\sigma(W^hh+W^ix)$，即当前状态是由历史状态和输入共同决定的
- $y=\sigma(W^oh^{'})$，即输出一般是当前状态的一个**维度映射**后通过softmax得到需要的数据



:star2: **代码解析**

模型定义：

```py
import torch
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
 
        self.out = nn.Linear(32, 1)
 
    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
 
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time, :]))
        return torch.stack(outs, dim=1), h_state
```

