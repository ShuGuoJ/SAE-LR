import torch
from Model.module import GreedyEncoder
from torch import nn, optim
from torch.utils.data import DataLoader
from Loss import CrossEntropy
import numpy as np
from scipy.io import loadmat
import os
from HSIDataset import HSIDataset, DatasetInfo
from torch.utils.data import random_split
from visdom import Visdom
import argparse
from utils import weight_init
# 参数
EPOCH_PER_LAYER = 5000
LR = 1e-1
BATCHSZ = 256
NUM_WORKERS = 1
RATIO = 0.8
SEED = 971104
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(SEED)
datasetInfo = DatasetInfo()
SAVE_ROOT = 'encoder'
DATASET_NAME = None



# 模型训练
def train(model, train_loader, test_loader):
    model.to(DEVICE)
    num_layers = model.get_num_of_layer()
    viz = Visdom(port=17000)
    for i in range(1, num_layers+1):
        print('*'*5 + 'Layer_{}'.format(i) + '*'*5)
        # 可视化学习曲线
        viz.line([[0., 0.]], [0.], win='encoder_layer_{}'.format(i), opts={'title':'encoder_layer_{}'.format(i),
                                                                           'legend':['train_loss', 'eval_loss']})
        model.set_trainable_layer(i)
        # note: 交叉熵并无法最小化误差，其误差最小化为x = 0.5
        criterion = CrossEntropy()
        # 过滤掉不可训练参数
        trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = optim.Adam(iter(trainable_parameters), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH_PER_LAYER//2, gamma=0.1)
        criterion.to(DEVICE)
        for epoch in range(EPOCH_PER_LAYER):
            train_loss = []
            for step, ((x, neighbor_region), _) in enumerate(train_loader):
                if neighbor_region.ndim == 4:
                    neighbor_region = neighbor_region.reshape((neighbor_region.shape[0], -1))
                # 拼接原始光谱信息和邻域信息
                input = torch.cat([x, neighbor_region], -1)
                input = input.to(DEVICE)
                raw_input, out = model(input) #[batchsz, n_feautre]
                # 计算Loss
                loss = criterion(out, raw_input)
                train_loss.append(loss.item())
                #反向传播
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                # l2_norm = nn.utils.clip_grad_norm_(trainable_parameters, 10)
                optimizer.step()

                if step%50==0:
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    # print('Layer-{} epoch:{} batch:{} loss:{:.6f} lr:{} l2-norm:{}'.format(i, epoch, step, loss.item(), lr, l2_norm))
                    print('Layer-{} epoch:{} batch:{} loss:{:.6f} lr:{}'.format(i, epoch, step, loss.item(), lr))

            # 测试
            eval_loss = []
            with torch.no_grad():
                for j, ((x, neighbor_region), _) in enumerate(test_loader):
                    if neighbor_region.ndim == 4:
                        neighbor_region = neighbor_region.reshape((neighbor_region.shape[0], -1))
                    # 拼接原始光谱信息和邻域信息
                    input = torch.cat([x, neighbor_region], -1)
                    input = input.to(DEVICE)
                    raw_input, out = model(input)
                    loss = criterion(out, raw_input)
                    # 可视化向量
                    if j==0:
                        num = 5
                        raw_vector, re_vector = raw_input[:num], out[:num]
                        length = raw_vector.shape[1]
                        for k in range(num):
                            viz.line(torch.stack([raw_vector[k], re_vector[k]], dim=0).T, list(range(length)), win='layer{}_sample{}'.format(i, k),
                                     opts={'title':'layer{}_sample{}'.format(i, k),
                                           'legend':['encoder', 'decoder']})

                    eval_loss.append(loss.item())
            train_mean_loss = float(np.mean(train_loss))
            eval_mean_loss = float(np.mean(eval_loss))
            print('Layer-{} epoch:{} train_loss: {:.6f} eval_loss:{:.6f}'.format(i, epoch, train_mean_loss, eval_mean_loss))
            viz.line([[train_mean_loss, eval_mean_loss]], [epoch], win='encoder_layer_{}'.format(i), update='append')
            scheduler.step()
            # 训练最后一层时，每经过50个epoch保存一个模型
            if i == num_layers and (epoch + 1) % 50 == 0:
                save_path = os.path.join(SAVE_ROOT, DATASET_NAME)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(model.state_dict(), os.path.join(save_path, 'stacked_auto_encoder_{}'.format(epoch)))
        print('*' * 5 + 'Finish' + '*' * 5)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Greedy layer-wise train')
    parse.add_argument('--epl', type=int, default=5000,
                       help='Train epoch for per layer')
    parse.add_argument('--name', type=str, default='PaviaU',
                       help='The name of dataset')
    parse.add_argument('--lr', type=float, default=1e-1,
                       help='Learning rate')
    parse.add_argument('--workers', type=int, default=16,
                       help='The num of workers')
    args = parse.parse_args()

    EPOCH_PER_LAYER = args.epl
    DATASET_NAME = args.name
    LR = args.lr
    NUM_WORKERS = args.workers
    # 模型、数据预处理
    info = DatasetInfo.info[DATASET_NAME]
    root = './data/{}'.format(DATASET_NAME)
    data = loadmat(os.path.join(root, '{}.mat'.format(DATASET_NAME)))[info['data_key']]
    label = loadmat(os.path.join(root, '{}_gt.mat'.format(DATASET_NAME)))[info['label_key']]
    data, label = data.astype(np.float32), label.astype(np.int)
    dataset = HSIDataset(data, label, patchsz=5, n_components=4)

    train_size = int(len(dataset) * RATIO)
    train_dataset, eval_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCHSZ, num_workers=NUM_WORKERS, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCHSZ, num_workers=NUM_WORKERS)

    model = GreedyEncoder(info['band']+5*5*4, info['hidden_size'])
    model.apply(weight_init)
    train(model, train_loader, eval_loader)

