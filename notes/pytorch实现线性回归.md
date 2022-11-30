# 线性回归pytorch流程

## 导入包

```python
%matplotlib inline
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)
```

## 处理数据

### 导入数据

```python
train_data = pd.read_csv('.csv')
test_data = pd.read_csv('.csv')
```

### 分离特征和标签

```python
# 拼接 axis=0表示按行  axis=1表示按列
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]), axis=0)
```

### 数据预处理(标准化数值+one hot 非数值)

```python
# 获取特征为数值的样本不是数值的索引
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 标准化数值
all_features[numeric_features] = all_features[numeric_features].apply(
   lambda x: (x - x.mean()) / (x.std()))
# std标准差
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# lambda 定义个变量 ： 一个函数返回到定义的变量中
# apply(func,*args,**kwargs)  当一个函数的参数存在于一个元组或者一个字典中时，
# 用来间接的调用这个函数，并将元组或者字典中的参数按照顺序传递给参数
# 有时候，函数的参数可能是DataFrame中的行或者列 axis=0表示行 axis=0表示列

# one hot encode非数值
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
# get_dummies 是利用pandas实现one hot encode的方式。
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

### 将数据转变为tensor

```python
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.'label'.values, dtype=torch.float).view(-1, 1)
# view 改变形状，-1表示原来形状
```

## 定义模型

```python
# 定义损失函数
loss = torch.nn.MSELoss()
# 初始化网络（线性网络，初始化所有参数）
def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net
```

### 定义对数平方根误差

```python
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()
```

### 定义训练函数

```python
# 定义训练参数（训练数据集，定义随机批量，计算损失函数）
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # 定义数据集（特征加标签）
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    # 定义随机批量 shuffle=True表示随机 batch_size表示批量大小
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 定义优化算法（这里使用了Adam优化算法，参数，学习了，权重）
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    net = net.float()
    # 定义训练流程num_epochs训练次数
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())# 损失函数
            optimizer.zero_grad()# 梯度清零
            l.backward() # 计算反向传播
            optimizer.step() # 进行参数优化
        train_ls.append(log_rmse(net, train_features, train_labels))# 计算对数平方
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

### 定义K折交叉验证

```python
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1  # 如果assert中为false则直接出发异常，无需运行所有程序后再出发
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)# 将验证集切出来
        X_part, y_part = X[idx, :], y[idx]
        if j == i:# 根据i来决定那一部分作为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:#剩下来的如果训练集为空就加入训练集
            X_train, y_train = X_part, y_part
        else:# 将生剩下了的都加入训练集
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid
```

### 定义K折交叉训练

```python
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)# 选择训练集和验证集
        net = get_net(X_train.shape[1])# 确定训练网络
        #训练并计算损失函数
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # 画图
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k #返回平均损失函数
```

## 选择超参数开始训练

```python
# 定义超参数
k, num_epochs, lr, weight_decay, batch_size = 10, 200, 5, 0, 4
# 训练并统计最终损失
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))
```

![image-20220211164606855](C:\Users\13288\AppData\Roaming\Typora\typora-user-images\image-20220211164606855.png)

![image-20220211165112852](C:\Users\13288\AppData\Roaming\Typora\typora-user-images\image-20220211165112852.png)

![image-20220211165321498](C:\Users\13288\AppData\Roaming\Typora\typora-user-images\image-20220211165321498.png)