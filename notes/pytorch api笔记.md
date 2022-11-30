# 图像增强

```python
from torchvision import datasets, models, transforms
```

[transforms的二十二个方法（pytorch） - 百度文库 (baidu.com)](https://wenku.baidu.com/view/3c743f6ecc84b9d528ea81c758f5f61fb73628c3.html)

## transforms.Compose

负责线性串联其他图像操作

```python
transforms.Compose([transforms.RandomResizedCrop(224),
 					transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
```

## transforms.RandomResizedCrop()

```python
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
#长宽200 随机裁剪一个面积为原始面积10%到100%的区域 区域的宽高比从0.5到2之间随机取值
```

将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
默认scale=(0.08, 1.0)

```python
img = Image.open("./demo.jpg")
print("原图大小：",img.size)
data1 = transforms.RandomResizedCrop(224)(img)
print("随机裁剪后的大小:",data1.size)
data2 = transforms.RandomResizedCrop(224)(img)
data3 = transforms.RandomResizedCrop(224)(img)

plt.subplot(2,2,1),plt.imshow(img),plt.title("原图")
plt.subplot(2,2,2),plt.imshow(data1),plt.title("转换后的图1")
plt.subplot(2,2,3),plt.imshow(data2),plt.title("转换后的图2")
plt.subplot(2,2,4),plt.imshow(data3),plt.title("转换后的图3")
plt.show()
```

![image-20220218142132257](C:\Users\13288\AppData\Roaming\Typora\typora-user-images\image-20220218142132257.png)

## transforms.CenterCrop(224)

中心裁剪

```
transforms.CenterCrop(224)
```

## transforms.RandomHorizontalFlip()

水平线翻转

```
img = Image.open("./demo.jpg")
img1 = transforms.RandomHorizontalFlip()(img)
img2 = transforms.RandomHorizontalFlip()(img)
img3 = transforms.RandomHorizontalFlip()(img)

plt.subplot(2,2,1),plt.imshow(img),plt.title("原图")
plt.subplot(2,2,2), plt.imshow(img1), plt.title("变换后的图1")
plt.subplot(2,2,3), plt.imshow(img2), plt.title("变换后的图2")
plt.subplot(2,2,4), plt.imshow(img3), plt.title("变换后的图3")
plt.show()
```

![image-20220218142235557](C:\Users\13288\AppData\Roaming\Typora\typora-user-images\image-20220218142235557.png)

## transforms.RandomVerticalFlip()

同上操作的垂直线翻转

## transforms.ColorJitter()

亮度（`brightness`）、对比度（`contrast`）、饱和度（`saturation`）和色调（`hue`）

```python
transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0)
# 随机更改图像的亮度，随机值为原始图像的50%（ 1−0.5 ）到150%（ 1+0.5 ）之间
```

![image-20220304121834915](C:\Users\13288\AppData\Roaming\Typora\typora-user-images\image-20220304121834915.png)

我们可以随机更改图像的色调

```python
transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5)
```

![image-20220304121943727](C:\Users\13288\AppData\Roaming\Typora\typora-user-images\image-20220304121943727.png)

## transforms.ToTensor() 

将给定图像转为Tensor

```python
img = Image.open("./demo.jpg") img = transforms.ToTensor()(img) print(img)
```



## transforms.Normalize(）

```python 
img = Image.open("./demo.jpg")
img = transforms.ToTensor()(img)
img = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(img)
#该通道的每个值减去该通道的平均值，然后将结果除以该通道的标准差
print(img)
```

# torch数据操作

## torch.cat

1. 字面理解：torch.cat是将两个张量（tensor）拼接在一起，cat是concatenate的意思，即拼接，联系在一起

2. [(767条消息) PyTorch的torch.cat_my-GRIT的博客-CSDN博客_torch.cat](https://blog.csdn.net/qq_39709535/article/details/80803003)

   ```python
   C=torch.cat((A,B),0)#按维数0（行）拼接
   ```


## torch.repeat_interleave

```python
torch.repeat_interleave(input, repeats, dim=None) → Tensor

x = torch.tensor([1, 2, 3])
x.repeat_interleave(2)
tensor([1, 1, 2, 2, 3, 3])
# 传入多维张量，默认`展平`
y = torch.tensor([[1, 2], [3, 4]])
torch.repeat_interleave(y, 2)
tensor([1, 1, 2, 2, 3, 3, 4, 4])
# 指定维度
torch.repeat_interleave(y,3,0)
tensor([[1, 2],
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4],
        [3, 4]])
torch.repeat_interleave(y, 3, dim=1)
tensor([[1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4]])
# 指定不同元素重复不同次数
torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
tensor([[1, 2],
        [3, 4],
        [3, 4]])
```

1.input (类型：torch.Tensor)：输入张量
2.repeats（类型：int或torch.Tensor）：每个元素的重复次数。repeats参数会被广播来适应输入张量的维度
3.dim（类型：int）需要重复的维度。默认情况下，将把输入张量展平（flatten）为向量，然后将每个元素重复repeats次，并返回重复后的张量。



## unsqueeze()和squeeze()函数

unsqueeze()在指定位置增加一个维度

squeeze()在指定位置去掉一个维度

[(744条消息) 【学习笔记】pytorch中squeeze()和unsqueeze()函数介绍_flysky_jay的博客-CSDN博客_squeeze](https://blog.csdn.net/flysky_jay/article/details/81607289)



## torch.nn.init——初始化参数

1. Xavier 初始化

2. nn.init 中各种初始化函数

3. He 初始化

[(585条消息) pytorch系列 -- 9 pytorch nn.init 中实现的初始化函数 uniform, normal, const, Xavier, He initialization_墨氲的博客-CSDN博客_torch.nn.init.xavier_uniform_](https://blog.csdn.net/dss_dssssd/article/details/83959474)



## torch.save()&model.load_state_dict

torch.save()用法：保存模型参数

```python
torch.save(model.state_dict(), f'transformer_best.pth')
```

加载模型

```python 
model.load_state_dict(torch.load(f'transformer_best.pth'))
```



# 优化

## 解释型转编译型

通过使⽤ torch.jit.script 函数来转换模型，我们就有能⼒编译和优化多层感知机中的计算，而模型的 计算结果保持不变。

```python
net = torch.jit.script(net)
```

## 保存模型

```python
net.save('my_mlp')
```

## GPU并行计算



# 参考链接

[Pytorch：transforms的二十二个方法_东方佑-CSDN博客](https://blog.csdn.net/weixin_32759777/article/details/108022357)
[PyTorch教程-6：详解PyTorch中的transforms - 简书 (jianshu.com)](https://www.jianshu.com/p/ae695df39274)