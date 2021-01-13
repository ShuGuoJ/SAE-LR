import torch
from torch import nn

class CrossEntropy(nn.Module):
    def forward(self, pred, target):
        '''
        :param pred:[batchsz,bands]
        :param target: [batchsz,bands]
        :return: scale
        '''

        loss = target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8)
        mean_loss = torch.mean(torch.sum(loss, axis=-1))
        return -mean_loss

# criterion = CrossEntropy()
# pred = torch.tensor([[0.9]], dtype=torch.float)
# target = torch.tensor([[1.]], dtype=torch.float)
# print(criterion(pred, target))
# print(torch.log(torch.tensor(0.9 + 1e-8)))