import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
class HSIDataset(Dataset):
    def __init__(self, data, label, n_components=1, patchsz=1):
        '''
        :param data: [h, w, bands]
        :param label: [h, w]
        :param n_components: scale
        :param patchsz: scale
        '''
        super(HSIDataset, self).__init__()
        self.data = data # [h, w, bands]
        self.label = label # [h, w]
        self.patchsz = patchsz
        # 原始数据的维度
        self.h, self.w, self.bands = self.data.shape
        self.data_pca = self.reduce_dimension(data, n_components)
        self.data = self.Normalize(self.data)
        self.data_pca = self.Normalize(self.data_pca)
        # self.get_mean()
        # # 数据中心化
        # self.data -= self.mean
        self.data_pca = self.addMirror(self.data_pca)

    # 计算投影矩阵
    def reduce_dimension(self, data, n_components):
        h, w, bands = data.shape
        data = data.reshape((h*w, bands))
        pca = PCA(n_components, whiten=True)
        data = pca.fit_transform(data)
        return data.reshape((h, w, -1))

    # 数据归一化
    def Normalize(self, data):
        h, w, c = data.shape
        data = data.reshape((h * w, c))
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0)
        data = data.reshape((h, w, c))
        return data

    # 添加镜像
    def addMirror(self, data):
        dx = self.patchsz // 2
        h, w, bands = data.shape
        mirror = None
        if dx != 0:
            mirror = np.zeros((h + 2 * dx, w + 2 * dx, bands))
            mirror[dx:-dx, dx:-dx, :] = data
            for i in range(dx):
                # 填充左上部分镜像
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # 填充右下部分镜像
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
        return mirror

    def __len__(self):
        return self.h * self. w

    def __getitem__(self, index):
        '''
        :param index:
        :return: 元素光谱信息， 元素的空间信息， 标签
        '''
        l = index // self.w
        c = index % self.w
        # 领域
        neighbor_region = self.data_pca[l:l + self.patchsz, c:c + self.patchsz, :]
        # 中心像素的光谱
        spectra = self.data[l, c]
        # 类别
        target = self.label[l, c] - 1
        return (torch.tensor(spectra, dtype=torch.float32), torch.tensor(neighbor_region, dtype=torch.float32)), \
        torch.tensor(target, dtype=torch.long)

class HSIDatasetV1(HSIDataset):
    def __init__(self, data, label, n_components=1, patchsz=1):
        super().__init__(data, label, n_components, patchsz)
        self.sampleIndex = list(zip(*np.nonzero(self.label)))

    def __len__(self):
        return len(self.sampleIndex)

    def __getitem__(self, index):
        l, c = self.sampleIndex[index]
        spectra = self.data[l, c]
        neighbor_region = self.data_pca[l:l + self.patchsz, c:c + self.patchsz, :]
        target = self.label[l, c] - 1
        return (torch.tensor(spectra, dtype=torch.float32), torch.tensor(neighbor_region, dtype=torch.float32)), \
                torch.tensor(target, dtype=torch.long)

class DatasetInfo(object):
    info = {'PaviaU': {
        'data_key': 'paviaU',
        'label_key': 'paviaU_gt',
        'hidden_size': 60,
        'band': 103
    },
        'Salinas': {
            'data_key': 'salinas_corrected',
            'label_key': 'salinas_gt',
            'hidden_size': 20,
            'band': 204
        },
        'KSC': {
            'data_key': 'KSC',
            'label_key': 'KSC_gt',
            'hidden_size': 20,
            'band': 176
    },  'Houston':{
            'data_key': 'Houston',
            'label_key': 'Houston2018_gt'
    }}


# from scipy.io import loadmat
# m = loadmat('data/KSC/KSC.mat')
# data = m['KSC'].astype(np.float32)
# m = loadmat('data/KSC/KSC_gt.mat')
# label = m['KSC_gt'].astype(np.int32)
# dataset = HSIDataset(data, label, patchsz=5, n_components=4)
# (spectra, neighbor_region), label = dataset[0]
# print(spectra.shape)
# print(neighbor_region.shape)
