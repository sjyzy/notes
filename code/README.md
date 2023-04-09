<!-- <link rel="stylesheet" type="text/css" href="/themes/newsprint.css"> -->

# code
## 数据处理

### 融合数据集（yolo标签版）

```python
import os
import shutil
import uuid
def merge_datasets(dataset1_path, dataset2_path, output_path):
    a = 0
    for dataset_path in [dataset1_path, dataset2_path]:
        for split in ['train', 'val']:
            for data_type in ['images', 'labels']:
                src_dir = os.path.join(dataset_path, split, data_type)
                dest_dir = os.path.join(output_path, split, data_type)
                os.makedirs(dest_dir, exist_ok=True)

                for file in os.listdir(src_dir):
                    a += 1
                    file_base, file_ext = os.path.splitext(file)
                    new_file_base = str(a).zfill(7)
                    new_file = f"{new_file_base}{file_ext}"
                    src_file = os.path.join(src_dir, file)
                    dest_file = os.path.join(dest_dir, new_file)
                    shutil.copy(src_file, dest_file)

                    # Maintain one-to-one correspondence between images and labels
                    if data_type == 'images':
                        label_ext = '.txt'
                        src_label = os.path.join(src_dir.replace('/images', '/labels'), f"{file_base}{label_ext}")
                        dest_label = os.path.join(dest_dir.replace('/images', '/labels'), f"{new_file_base}{label_ext}")
                        print(src_label)  # Debug: print src_label
                        print(dest_label)  # Debug: print dest_label
                        shutil.copy(src_label, dest_label)


# Define the paths to the two datasets and the output directory
dataset1_path = "/nas_data/SJY/code/yolov7/person_phone"
dataset2_path = "/nas_data/SJY/code/yolov7/smoke"
output_path = "/nas_data/SJY/code/yolov7/dataset"

# Merge the two datasets into one
merge_datasets(dataset1_path, dataset2_path, output_path)

print("Merged the datasets and created train and test image list files.")
```

### 生成图片本地绝对路径文件（yolo标签）

```python
import os

def save_image_paths(input_dir, output_file):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    with open(output_file, 'w') as f_out:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    abs_path = os.path.abspath(os.path.join(root, file))
                    f_out.write(abs_path + '\n')

if __name__ == "__main__":
    input_dir = '/nas_data/SJY/code/yolov7/dataset/images/val'
    output_file = '/nas_data/SJY/code/yolov7/dataset/val_list.txt'

    if not os.path.exists(input_dir):
        print(f"The input directory '{input_dir}' does not exist.")
    else:
        save_image_paths(input_dir, output_file)
        print(f"Image paths successfully saved in '{output_file}'.")
```

### 更改标签编号（yolo标签）

```python
import os

def convert_label(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = line.strip().split()

            if data[0] == '0':
                data[0] = '2'

            f_out.write(' '.join(data) + '\n')

def process_directory(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"The input directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            convert_label(input_file, output_file)

if __name__ == "__main__":
    input_dir = '/nas_data/SJY/code/yolov7/smoke/val/labels_ori'
    output_dir = '/nas_data/SJY/code/yolov7/smoke/val/labels'

    process_directory(input_dir, output_dir)
    print(f"Labels successfully converted and saved in '{output_dir}'.")
```

### CUB数据格式

[CUB数据格式](/notes/CUB%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F.md)  

### 切割数据集

[切割数据集](/notes/%E5%88%87%E5%89%B2%E6%95%B0%E6%8D%AE%E9%9B%86.md)  

### voc转coco
[voc转coco](/notes/voc%E8%BD%ACcoco.md)  

### 图片数据集转txt流程

[图片数据集转txt流程](/notes/%E5%9B%BE%E7%89%87%E6%95%B0%E6%8D%AE%E9%9B%86%E8%BD%ACtxt%E6%B5%81%E7%A8%8B.md)  
### coco选择指定标签转yolo

```python
import os
import json
import shutil
from pycocotools.coco import COCO

def convert_to_yolo_format(bbox, img_width, img_height):
    x, y, width, height = bbox
    x_center = x + width / 2
    y_center = y + height / 2
    return [
        x_center / img_width,
        y_center / img_height,
        width / img_width,
        height / img_height,
    ]

# Define the paths to the COCO dataset
coco_images_path = "/nas_data/COMMON/dataset/coco/val2017"
annotations_path = "/nas_data/COMMON/dataset/coco/annotations/instances_val2017.json"
# annotations_path = "/nas_data/COMMON/dataset/coco/annotations/instances_train2017.json"


# Define the output directories
output_images_path = "/nas_data/SJY/code/yolov7/dataset/val/image"
output_labels_path = "/nas_data/SJY/code/yolov7/dataset/val/annotations"

# Create output directories if they do not exist
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

# Load COCO API with the annotations file
coco = COCO(annotations_path)

# Get the category ids for cell phones and humans
cell_phone_category_id = coco.getCatIds(catNms=["cell phone"])
human_category_id = coco.getCatIds(catNms=["person"])

# Get all the image ids containing cell phones or humans
cell_phone_image_ids = coco.getImgIds(catIds=cell_phone_category_id)
human_image_ids = coco.getImgIds(catIds=human_category_id)
combined_image_ids = list(set(cell_phone_image_ids + human_image_ids))

# Create YOLO format labels for images containing cell phones or humans
for img_id in combined_image_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_filename = img_info["file_name"]
    img_width = img_info["width"]
    img_height = img_info["height"]

    # Get the annotations for the current image
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=(cell_phone_category_id + human_category_id), iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # Save the annotations in YOLO format
    with open(os.path.join(output_labels_path, f"{img_id}.txt"), "w") as f:
        for ann in anns:
            yolo_bbox = convert_to_yolo_format(ann["bbox"], img_width, img_height)
            category_id = ann["category_id"]
            if category_id == cell_phone_category_id[0]:
                yolo_label = 0
            elif category_id == human_category_id[0]:
                yolo_label = 1
            f.write(f"{yolo_label} {' '.join(map(str, yolo_bbox))}\n")

    # Copy the image to the output directory
    shutil.copy(os.path.join(coco_images_path, img_filename), os.path.join(output_images_path, img_filename))

print(f"Extracted {len(combined_image_ids)} cell phone and human images and converted annotations to YOLO format.")

```

### 统一文件名位数（用于图片指定标签）

```python
import os

def pad_and_rename_files(directory_path):
    for file in os.listdir(directory_path):
        file_name, file_ext = os.path.splitext(file)

        # Check if the file name is made up of digits only
        if file_name.isdigit():
            # Pad the file name with zeros to make it 12 digits
            new_file_name = file_name.zfill(12)
            new_file = f"{new_file_name}{file_ext}"

            # Rename the file
            src_file = os.path.join(directory_path, file)
            dest_file = os.path.join(directory_path, new_file)
            os.rename(src_file, dest_file)

# Specify the directory containing the files to be renamed
directory_path = "/nas_data/SJY/code/yolov7/person_phone/val/labels"

# Rename the files
pad_and_rename_files(directory_path)

print("Renamed files with 12-digit padded names.")
```




## 图像处理
### 将图像背景变透明

```python
# 输入图片路径和输出图片路径
input_path = "/_media/logo.png"
output_path = "/_media/logo_transparency.png"

# 打开图片并将背景变为透明
with Image.open(input_path) as im:
    im = im.convert("RGBA")
    data = im.getdata()

    newData = []
    for item in data:
        # 将背景色的像素点变为透明
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    im.putdata(newData)

    # 保存处理后的图片
    im.save(output_path)
```


