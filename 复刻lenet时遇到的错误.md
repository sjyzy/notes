# RuntimeError：stack expects each tensor to be equal size, but got [3, 295, 300] at entry 0 and [3, 229, 300] at entry 1

num_workers改为0即可

```python
test_iter = DataLoader(dataset=test_data, batch_size=6, shuffle=False,num_workers=0)
```

# RuntimeError: stack expects each tensor to be equal size, but got [3, 295, 300] at entry 0 and [3, 229, 300] at entry 1

这是图片预处理的问题

```python
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



# mat1 and mat2 shapes cannot be multiplied (6x8464 and 400x120)

网络参数与实际输入量不符合

通过运行

```python
X = torch.rand(size=(3, 3, 100, 100), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

来追踪每一层的输出大小