# 导入包

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

# 创建数据集（和lenet一样）

```python
def default_loader(path):# 默认图片加载方法
        return Image.open(path).convert('RGB')
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

#实例化上面定义的数据集，并且加载上面定义的transform
train_data=MyDataset(txt=r'D:\dataset\Fruit\train.txt', transform=train_transforms)
test_data = MyDataset(txt=r'D:\dataset\Fruit\text.txt', transform=text_transforms)
#然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，注意，loader的长度是有多少个batch，所以和batch_size有关
train_iter = DataLoader(dataset=train_data, batch_size=10, shuffle=True,num_workers=0)
test_iter = DataLoader(dataset=test_data, batch_size=6, shuffle=False,num_workers=0)
# 查看数据集信息
print('num_of_trainData:', len(train_data))
print('num_of_testData:', len(test_data))
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

# 创建inception块

```python
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
```

# 创建网路



```python
b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 3))

X = torch.rand(size=(3, 3, 100, 100))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

# 训练

``` python
lr, num_epochs, batch_size = 0.05, 100, 128
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```





