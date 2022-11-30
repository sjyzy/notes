CUB数据格式

CUB_200_2011
├── attributes
├── images # 存放所有图片类的图片
├── parts
├── bounding_boxes.txt
├── classes.txt
├── image_class_labels.txt
├── images.txt
├── README
└── train_test_split.txt

获取所有images.txt、classes.txt、image_class_labels.txt 和 train_test_split.txt

```python
import os
import shutil

images_path = "./images" #文件夹目录
images = os.listdir(images_path) #得到文件夹下的所有文件名称

def get_all_txt():
    """
    获取 classes.txt、images.txt、image_class_labels.txt和train_test_split.txt
    """
    class_txt = open('classes.txt', 'w')
    train_test_split_txt = open('train_test_split.txt', 'w')
    image_class_labels_txt = open('image_class_labels.txt', 'w')
    images_txt = open('images.txt', 'w')
    images_ids = 0
    for class_ids, class_name in enumerate(images): #遍历文件夹
        class_txt.write(str(class_ids+1)+  ' ' + class_name + '\n')
        class_path = os.path.join(images_path, class_name)
        class_image = os.listdir(class_path)
        for image_name in class_image:
            image_path = os.path.join(class_path, image_name)
            images_ids = images_ids + 1
            images_txt.write(str(images_ids) + ' ' + image_path[9:] + '\n')
            image_class_labels_txt.write(str(images_ids) + ' ' + str(class_ids+1)+'\n')
            if images_ids%4 == 1:
                train_test_split_txt.write(str(images_ids) +' 0\n')
            else:
                train_test_split_txt.write(str(images_ids) +' 1\n')
            print(image_name)

get_all_txt()
```

