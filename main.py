import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
"""

Hinton最新研究 Forward Forward 网络 中文翻译版本，可以看这篇文章：
    https://www.163.com/dy/article/HNLP6E5N0511831M.html
    
"""

"""
描述:
   mnist 数据加载(如果本地没有数据从第3方下载，请保证本地的网络流畅或用全局代理)
参数：
   train_batch_size：训练时每次的batch取的数据条数
   test_batch_size：测试时每次的batch取的数据条数
返回结果：
   train_loader: 训练数据加载器
   test_loader: 测试数据加载器
"""
def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    # 数据变换器
    transform = Compose([
        ToTensor(), # 转化数据为torch.Tensor类型
        Normalize((0.1307,), (0.3081,)),   # 对 图像的进行正则化，0.1307 为平均值，0.3081 为标准差
        Lambda(lambda x: torch.flatten(x))]) # 对图像数据进行展平（说白了，多维度变换为1维)

    # 加载训练数据，并 随机取数据
    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    # 加载测试数据，并 随机取数据
    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


# 做掩码操作，可以认为是对图像添加噪声
def overlay_y_on_x(x, y):
    # 复制x
    x_ = x.clone()
    # 对每个图像的前10个像素值，设置为0.0
    x_[:, :10] *= 0.0
    # 对每个图像的第y个像素值，设置为x的最大值
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):

    """
    描述:
       初始化网络或model
    参数：
       dims：维度数组，如：[768,500]
    """
    def __init__(self, dims):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for d in range(len(dims) - 1):
            layer = Layer(dims[d], dims[d + 1])
            self.layers = self.layers.append(layer)

    """
    描述：
       推理方法，相当与__call__ 方法
    参数：
        x 为图像数据，维度为 (bs,in_features)
    结果：
        计算后的结果,维度为 (bs,out_features)
    """
    def predict(self, x):
        # 好的标签数组
        goodness_per_label = []
        # 10 表示标签个数,label 为 0 到 9的某个数
        for label in range(10):
            # 对 x 添加噪声
            h = overlay_y_on_x(x, label)
            goodness = []
            # 各层预测结果
            for layer in self.layers:
                # 层预测输出
                h = layer(h)
                # h为预测值，对预测值平方和维度1平均化，且 添加到结果，维度为 (50000,)
                goodness += [h.pow(2).mean(1)]
            # 汇总多层的结果，并扩展维度
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        # 用torch.cat函数转换为torch.Tensor 数据, 且 保持维度不变
        goodness_per_label = torch.cat(goodness_per_label, 1)
        # 求标签的最大概率，goodness_per_label 维度为 (bs,10)
        return goodness_per_label.argmax(1)

    """
    描述：
        分层训练，每个层单独训练
    参数：
        x_pos : 正样本数据，维度为 （bs，图像维度1d)
        x_neg : 负样本数据，维度为 （bs，图像维度1d)
    结果：
        无
    """
    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    """
    描述:
       线性层
    参数：
       in_features：输入维度
       out_features：输出维度
       bias：是否开启bias
       device：运行设备信息，如：cpu，cuda，mps等
       dtype：数据类型
    """
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        # 激活函数
        self.relu = torch.nn.ReLU()
        # adam 优化器
        self.opt = Adam(self.parameters(), lr=0.03)
        # 阀值，超参数，用于控制loss值的方向
        self.threshold = 2.0
        # 每层的运行次数
        self.num_epochs = 1000

    """
    描述：
       推理方法，相当与__call__ 方法
    参数：
        x 为图像数据，维度为 (bs,in_features)
    结果：
        计算后的结果,维度为 (bs,out_features)
    """
    def forward(self, x):
        # x 正则化
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    """
    描述：
        训练
    参数：
        x_pos : 正样本数据，维度为 （bs，图像维度1d)
        x_neg : 负样本数据，维度为 （bs，图像维度1d)
    结果：
        x_pos_re 计算后的结果,维度为 (bs,out_features)
        x_neg_re 计算后的结果,维度为 (bs,out_features)
    """
    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            # 正样本计算，并计算幂和平均值
            g_pos = self.forward(x_pos).pow(2).mean(1)
            # 负样本计算，并计算幂和平均值
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            # 计算loss
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            # 优化器中的梯度清零
            self.opt.zero_grad()
            # 计算梯度
            #
            loss.backward()
            # 优化器步进且更新权重
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

if __name__ == "__main__":
    # 设置随机数的种子，用于复现model的结果
    torch.manual_seed(1234)
    # 加载数据
    train_loader, test_loader = MNIST_loaders()

    #第1层的输入维度为768，输出500；2层依次推
    net = Net([784, 500, 500])
    # 获取训练数据，x和y， x 的 维度 (50000,768), y的维度为 (50000,1)
    x, y = next(iter(train_loader))
  #  x, y = x.cuda(), y.cuda()
    # 对正样本添加mask
    x_pos = overlay_y_on_x(x, y)
    # 对负样本随机添加mask
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    # 训练
    net.train(x_pos, x_neg)
    # 训练时的错误率
    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    # 加载测试数据
    x_te, y_te = next(iter(test_loader))
    #x_te, y_te = x_te.cuda(), y_te.cuda()
    # 测试时的错误率
    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
