

# 导入所需包

```python
import os
import time
import torch
from torchvision import datasets, models, transforms
from torch import nn, optim
from d2l import torch as d2l
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(device)
```

```
1.10.1
cuda
```

# 设计网络

网络根据数据集来设计

```python
net = nn.Sequential(
    # 参数一：输入通道，参数二：输出通道
    nn.Conv2d(3, 6, kernel_size=5, padding=2), 
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(6, 16, kernel_size=5), 
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 转换为线性层的时候需要计算大小，下面遍历出来的结果可以看到第二个卷积层后的输出大小
    # 这里是8464
    nn.Flatten(),
    nn.Linear(8464, 2000), nn.Sigmoid(),
    nn.Linear(2000, 100), nn.Sigmoid(),
    nn.Linear(100, 3))

X = torch.rand(size=(3, 3, 100, 100), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

```
Conv2d output shape: 	 torch.Size([3, 6, 100, 100])
Sigmoid output shape: 	 torch.Size([3, 6, 100, 100])
AvgPool2d output shape: 	 torch.Size([3, 6, 50, 50])
Conv2d output shape: 	 torch.Size([3, 16, 46, 46])
Sigmoid output shape: 	 torch.Size([3, 16, 46, 46])
AvgPool2d output shape: 	 torch.Size([3, 16, 23, 23])
Flatten output shape: 	 torch.Size([3, 8464])
Linear output shape: 	 torch.Size([3, 2000])
Sigmoid output shape: 	 torch.Size([3, 2000])
Linear output shape: 	 torch.Size([3, 100])
Sigmoid output shape: 	 torch.Size([3, 100])
Linear output shape: 	 torch.Size([3, 3])
```

# 准备数据

## 定义数据集

```python
# 对于彩色图像，不管其图像格式是PNG，还是BMP，或者JPG，
# 在PIL中，使用Image模块的open()函数打开后，返回的图像对象的模式都是“RGB”。
# 而对于灰度图像，不管其图像格式是PNG，还是BMP，或者JPG，打开后，其模式为“L”。
def default_loader(path):# 默认图片加载方法
        return Image.open(path).convert('RGB')
```

Dataset类是Pytorch中图像数据集中最为重要的一个类，也是Pytorch中所有数据集加载类中应该继承的父类。其中父类中的两个私有成员函数必须被重载，否则将会触发错误提示，其中__len__应该返回数据集的大小，而__getitem__应该编写支持数据集索引的函数，getitem接收一个index，然后返回图片数据和标签，这个index通常指的是一个list的index，这个list的每个元素就包含了图片数据的路径和标签信息。制作这个list，通常的方法是将图片的路径和标签信息存储在一个txt中，然后从该txt中读取。

那么读取自己数据的基本流程就是：

1.制作存储了图片的路径和标签信息的txt
2.将这些信息转化为list，该list每一个元素对应一个样本
3.通过getitem函数，读取数据和标签，并返回数据和标签

```python
#首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，
#然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset,self).__init__()#对继承自父类的属性进行初始化
        fh = open(txt, 'r')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        for line in fh:#迭代该列表，按行循环txt文本中的内容
            line = line.strip('\n')
            line = line.rstrip()# 删除 本行string 字符串末尾的指定字符
            words = line.split()#用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格，分割路径和标签
            if len(words)>2:# 防止路径中有空格，如果有空格就把前面的放在words[0]中
                words[0] = str((words[0]))+' '+str((words[1]))
                words[1] = words[2]
            #把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定 
            imgs.append((words[0],int(words[1])))# words[0]表示路径，words[1]表示标签
            
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):#这个方法必须要有，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]#fn是图片path,fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)# 按照路径读取图片
        if self.transform is not None:# 为了防止dataset代码冗余，一般把图像预处理单独操作
            img = self.transform(img)#数据标签转换为Tensor
        return img,label #return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
```

## 图像预处理

```python
#图像预处理
#transforms.Compose串联多个图片变换操作，这里是串联了RandomResizedCrop和ToTensor
train_transforms = transforms.Compose([
    #RandomResizedCrop，将给定图像随机裁剪为不同的大小和宽高比，
    #然后缩放所裁剪得到的图像为制定的大小；
    #即先随机采集，然后对裁剪得到的图像缩放为同一大小，默认scale=(0.08, 1.0)
    transforms.RandomResizedCrop((100,100)),
    transforms.ToTensor(),
])

text_transforms = transforms.Compose([
    transforms.RandomResizedCrop((100,100)),
    transforms.ToTensor(),
])
```

## 加载数据集

```python
#实例化上面定义的数据集，并且加载上面定义的transform
train_data=MyDataset(txt=r'D:\dataset\Fruit\train.txt', transform=train_transforms)
test_data = MyDataset(txt=r'D:\dataset\Fruit\text.txt', transform=text_transforms)
#然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，注意，loader的长度是有多少个batch，所以和batch_size有关
train_iter = DataLoader(dataset=train_data, batch_size=6, shuffle=True,num_workers=0)
test_iter = DataLoader(dataset=test_data, batch_size=6, shuffle=False,num_workers=0)
# 查看数据集信息
print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

# 定义评估函数

```python
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

# 定义训练函数

```python
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):#初始化权重
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)# xavier_uniform保证输出和输入的weight方差是一样的，防止一开始梯度爆炸和消失
    #初始化优化函数，损失函数        
    net.apply(init_weights) # 对每一层参数都进行初始化
    print('training on', device)
    net.to(device)# 加载到gpu中
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)# 定义优化函数
    loss = nn.CrossEntropyLoss()# 定义损失函数
    #初始化画图
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    #开始迭代训练
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()# 计时
            optimizer.zero_grad()# 梯度清零
            X, y = X.to(device), y.to(device) #加载到gpu中
            y_hat = net(X) 
            l = loss(y_hat, y)# 赋值损失函数
            l.backward()# 反向传播
            optimizer.step()# 优化权重
            with torch.no_grad():# 画图（不计算梯度）
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    #打印精度和损失曲线
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

# 训练

```python
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

# 参考文献

[(769条消息) pytorch学习（一）利用pytorch训练一个最简单的分类器-------（基于CIFAR10数据集）的句句讲解_sinat_42239797的博客-CSDN博客](https://blog.csdn.net/sinat_42239797/article/details/90373471)

[(769条消息) pytorch 加载自己的数据集_shunshune的博客-CSDN博客_pytorch加载自己的数据集](https://blog.csdn.net/shunshune/article/details/89316572)

[(769条消息) Pytorch学习（三）定义自己的数据集及加载训练_sinat_42239797的博客-CSDN博客_pytorch自定义数据集](https://blog.csdn.net/sinat_42239797/article/details/90641659)
