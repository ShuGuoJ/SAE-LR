import torch
from torch import nn
from torch.nn import functional as F
import os

# 编码器
class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers=5):
        super().__init__()
        self.feature = nn.Sequential()
        for i in range(1, num_layers+1):
            self.feature.add_module('layer_{}'.format(i), nn.Linear(in_channel if i==1 else out_channel, out_channel))
            self.feature.add_module('sigmoid_{}'.format(i), nn.Sigmoid())

    def forward(self, x):
        return self.feature(x)


# 贪婪预训练编码器
class GreedyEncoder(Encoder):
    def forward(self, x):
        assert hasattr(self, 'n_th_layer')
        raw_input = x if self.n_th_layer==1 else self.feature[:2*(self.n_th_layer-1)](x)
        code = self.feature[2*(self.n_th_layer-1):2*self.n_th_layer](raw_input)
        x_hat = torch.mm(code, self.feature[(self.n_th_layer-1)*2].weight) + self.b_decoder
        return raw_input, torch.sigmoid(x_hat)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        assert hasattr(self, 'n_th_layer')
        for p in self.feature[(self.n_th_layer-1)*2].parameters():
            p.requires_grad_()

    def set_trainable_layer(self, num):
        assert num>0 and num<=len(self.feature)//2
        setattr(self, 'n_th_layer', num)
        index = int((num-1)*2)
        self.freeze()
        self.unfreeze()
        device = self.feature[index].weight.device
        setattr(self, 'b_decoder', nn.Parameter(torch.zeros(self.feature[index].weight.shape[1], dtype=torch.float,
                                                            device=device, requires_grad=True)))

    def get_num_of_layer(self):
        return len(self.feature)//2

    def remove(self):
        delattr(self, 'b_decoder')
        delattr(self, 'n_th_layer')


# 测试encoder
# encoder = Encoder(174, 20)
# input = torch.rand((2,174),dtype=torch.float)
# print(encoder)
# out = encoder(input)
# print(out.shape)

# 测试GreedyEncoder
# net = GreedyEncoder(174, 20)
# net.set_layer(2)
# print(net)
# print(net.b_decoder)
# net.freeze()
# n = 0
# for p in net.parameters():
#     n += 1
# print(n)
# net = GreedyEncoder(2, 1)
# input = torch.FloatTensor([[1, 2]])
# print(net.feature[0].weight)
# net.set_layer(1)
# print(net.b_decoder.requires_grad)
# out = net(input)
# out = out.sum()
# out.backward()
# print(net.b_decoder.grad)
# print(net.feature[0].weight.grad)

# 测试GreedyEncoder
# net = GreedyEncoder(3, 2)
# net.set_trainable_layer(2)
# input = torch.rand((1,3))
# raw_input, out = net(input)
# print(torch.equal(input, raw_input))

