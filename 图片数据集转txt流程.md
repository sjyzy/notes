```python
import os

a = 0
while a < 3:  # 类别数
    dir = r'D:\dataset\Fruit'+'\\'+str(a)+'\\'  # 图片文件的地址
    label = a
    # os.listdir的结果就是一个list集，可以使用list的sort方法来排序。如果文件名中有数字，就用数字的排序
    files = os.listdir(dir)  # 列出dirname下的目录和文件
    files.sort()  # 排序
    train = open(r'D:\dataset\Fruit\train.txt', 'a')
    text = open(r'D:\dataset\Fruit\text.txt', 'a')
    i = 1
    for file in files:
        if i < 75:   # 训练集大小
            fileType = os.path.split(file)  # os.path.split()：按照路径将文件名和路径分割开
            if fileType[1] == '.txt':
                continue
            name = str(dir) + file + ' ' + str(int(label)) + '\n'
            train.write(name)
            i = i + 1
            print(i)
        else:   # 测试集大小
            fileType = os.path.split(file)
            if fileType[1] == '.txt':
                continue
            name = str(dir) + file + ' ' + str(int(label)) + '\n'
            text.write(name)
            i = i + 1
            print(i)
    text.close()
    train.close()
    print(a)
    a = a + 1

```

