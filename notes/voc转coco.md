```python
import xml.etree.ElementTree as ET
import os
import json
# from icecream import ic
coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = [{'supercategory': 'none', 'id': 0, 'name': '2xist'}, {'supercategory': 'none', 'id': 1, 'name': '3nod'}, {'supercategory': 'none', 'id': 2, 'name': '4Skins'}, {'supercategory': 'none', 'id': 3, 'name': 'Absolut'}, {'supercategory': 'none', 'id': 4, 'name': 'ACN'}, {'supercategory': 'none', 'id': 5, 'name': 'ADCO'}, {'supercategory': 'none', 'id': 6, 'name': 'Addicted'}, {'supercategory': 'none', 'id': 7, 'name': 'adika'}, {'supercategory': 'none', 'id': 8, 'name': 'Adnams'}, {'supercategory': 'none', 'id': 9, 'name': 'ADOX'}, {'supercategory': 'none', 'id': 10, 'name': 'Adriatica'}, {'supercategory': 'none', 'id': 11, 'name': 'aeknil'}, {'supercategory': 'none', 'id': 12, 'name': 'akurra'}, {'supercategory': 'none', 'id': 13, 'name': 'alcan'}, {'supercategory': 'none', 'id': 14, 'name': 'Aprilia'}, {'supercategory': 'none', 'id': 15, 'name': 'Arette'}, {'supercategory': 'none', 'id': 16, 'name': 'atamel'}, {'supercategory': 'none', 'id': 17, 'name': 'attack'}, {'supercategory': 'none', 'id': 18, 'name': 'Audison'}, {'supercategory': 'none', 'id': 19, 'name': 'Aussie'}, {'supercategory': 'none', 'id': 20, 'name': 'Bazooka'}, {'supercategory': 'none', 'id': 21, 'name': 'BBK'}, {'supercategory': 'none', 'id': 22, 'name': 'BCP'}, {'supercategory': 'none', 'id': 23, 'name': 'BeautiFeel'}, {'supercategory': 'none', 'id': 24, 'name': 'Binatone'}, {'supercategory': 'none', 'id': 25, 'name': 'Blimpie'}, {'supercategory': 'none', 'id': 26, 'name': 'Bona'}, {'supercategory': 'none', 'id': 27, 'name': 'Breyers'}, {'supercategory': 'none', 'id': 28, 'name': 'BUGATTI'}, {'supercategory': 'none', 'id': 29, 'name': 'Cachantun'}, {'supercategory': 'none', 'id': 30, 'name': 'Cacharel'}, {'supercategory': 'none', 'id': 31, 'name': 'Calistoga'}, {'supercategory': 'none', 'id': 32, 'name': 'camlin'}, {'supercategory': 'none', 'id': 33, 'name': 'canadel'}, {'supercategory': 'none', 'id': 34, 'name': 'Carola'}, {'supercategory': 'none', 'id': 35, 'name': 'Carrows'}, {'supercategory': 'none', 'id': 36, 'name': 'Cheerios'}, {'supercategory': 'none', 'id': 37, 'name': 'Chocomel'}, {'supercategory': 'none', 'id': 38, 'name': 'Chopard'}, {'supercategory': 'none', 'id': 39, 'name': 'Cif'}, {'supercategory': 'none', 'id': 40, 'name': 'Colt'}, {'supercategory': 'none', 'id': 41, 'name': 'Condor'}, {'supercategory': 'none', 'id': 42, 'name': 'Connector'}, {'supercategory': 'none', 'id': 43, 'name': 'ERAL'}, {'supercategory': 'none', 'id': 44, 'name': 'FSL'}, {'supercategory': 'none', 'id': 45, 'name': 'Gabor'}, {'supercategory': 'none', 'id': 46, 'name': 'Gitzo'}, {'supercategory': 'none', 'id': 47, 'name': 'gumpert'}, {'supercategory': 'none', 'id': 48, 'name': 'HEC'}, {'supercategory': 'none', 'id': 49, 'name': 'Hermes'}, {'supercategory': 'none', 'id': 50, 'name': 'Hilleberg'}, {'supercategory': 'none', 'id': 51, 'name': 'Home Inn'}, {'supercategory': 'none', 'id': 52, 'name': 'Hoover'}, {'supercategory': 'none', 'id': 53, 'name': 'Hovis'}, {'supercategory': 'none', 'id': 54, 'name': 'Inventec'}, {'supercategory': 'none', 'id': 55, 'name': 'islabikes'}, {'supercategory': 'none', 'id': 56, 'name': 'Isosource'}, {'supercategory': 'none', 'id': 57, 'name': 'Joico'}, {'supercategory': 'none', 'id': 58, 'name': 'Kolynos'}, {'supercategory': 'none', 'id': 59, 'name': 'levante'}, {'supercategory': 'none', 'id': 60, 'name': 'lishen'}, {'supercategory': 'none', 'id': 61, 'name': 'Mars'}, {'supercategory': 'none', 'id': 62, 'name': 'mayinglong'}, {'supercategory': 'none', 'id': 63, 'name': 'Mazda'}, {'supercategory': 'none', 'id': 64, 'name': 'melatonin'}, {'supercategory': 'none', 'id': 65, 'name': 'molten'}, {'supercategory': 'none', 'id': 66, 'name': 'onida'}, {'supercategory': 'none', 'id': 67, 'name': 'oovoo'}, {'supercategory': 'none', 'id': 68, 'name': 'paralen'}, {'supercategory': 'none', 'id': 69, 'name': 'perdolan'}, {'supercategory': 'none', 'id': 70, 'name': 'pinarello'}, {'supercategory': 'none', 'id': 71, 'name': 'qq'}, {'supercategory': 'none', 'id': 72, 'name': 'rasti'}, {'supercategory': 'none', 'id': 73, 'name': 'razor'}, {'supercategory': 'none', 'id': 74, 'name': 'ribena'}, {'supercategory': 'none', 'id': 75, 'name': 'rolo'}, {'supercategory': 'none', 'id': 76, 'name': 'sachs'}, {'supercategory': 'none', 'id': 77, 'name': 'saimaza'}, {'supercategory': 'none', 'id': 78, 'name': 'scrabble'}, {'supercategory': 'none', 'id': 79, 'name': 'sdlg'}, {'supercategory': 'none', 'id': 80, 'name': 'shua'}, {'supercategory': 'none', 'id': 81, 'name': 'sixpoint'}, {'supercategory': 'none', 'id': 82, 'name': 'slazenger'}, {'supercategory': 'none', 'id': 83, 'name': 'steeden'}, {'supercategory': 'none', 'id': 84, 'name': 'stolichnaya'}, {'supercategory': 'none', 'id': 85, 'name': 'swix'}, {'supercategory': 'none', 'id': 86, 'name': 'taoranju'}, {'supercategory': 'none', 'id': 87, 'name': 'tapsin'}, {'supercategory': 'none', 'id': 88, 'name': 'titleist'}, {'supercategory': 'none', 'id': 89, 'name': 'toblerone'}, {'supercategory': 'none', 'id': 90, 'name': 'univox'}, {'supercategory': 'none', 'id': 91, 'name': 'vango'}, {'supercategory': 'none', 'id': 92, 'name': 'vico'}, {'supercategory': 'none', 'id': 93, 'name': 'weibo'}, {'supercategory': 'none', 'id': 94, 'name': 'wiesmann'}, {'supercategory': 'none', 'id': 95, 'name': 'winsor'}, {'supercategory': 'none', 'id': 96, 'name': 'yangshengtang'}, {'supercategory': 'none', 'id': 97, 'name': 'yonex'}, {'supercategory': 'none', 'id': 98, 'name': 'yunnanbaiyao'}, {'supercategory': 'none', 'id': 99, 'name': 'zoegas'}]

category_set = {'2xist': 0, '3nod': 1, '4Skins': 2, 'Absolut': 3, 'ACN': 4, 'ADCO': 5, 'Addicted': 6, 'adika': 7, 'Adnams': 8, 'ADOX': 9, 'Adriatica': 10, 'aeknil': 11, 'akurra': 12, 'alcan': 13, 'Aprilia': 14, 'Arette': 15, 'atamel': 16, 'attack': 17, 'Audison': 18, 'Aussie': 19, 'Bazooka': 20, 'BBK': 21, 'BCP': 22, 'BeautiFeel': 23, 'Binatone': 24, 'Blimpie': 25, 'Bona': 26, 'Breyers': 27, 'BUGATTI': 28, 'Cachantun': 29, 'Cacharel': 30, 'Calistoga': 31, 'camlin': 32, 'canadel': 33, 'Carola': 34, 'Carrows': 35, 'Cheerios': 36, 'Chocomel': 37, 'Chopard': 38, 'Cif': 39, 'Colt': 40, 'Condor': 41, 'Connector': 42, 'ERAL': 43, 'FSL': 44, 'Gabor': 45, 'Gitzo': 46, 'gumpert': 47, 'HEC': 48, 'Hermes': 49, 'Hilleberg': 50, 'Home Inn': 51, 'Hoover': 52, 'Hovis': 53, 'Inventec': 54, 'islabikes': 55, 'Isosource': 56, 'Joico': 57, 'Kolynos': 58, 'levante': 59, 'lishen': 60, 'Mars': 61, 'mayinglong': 62, 'Mazda': 63, 'melatonin': 64, 'molten': 65, 'onida': 66, 'oovoo': 67, 'paralen': 68, 'perdolan': 69, 'pinarello': 70, 'qq': 71, 'rasti': 72, 'razor': 73, 'ribena': 74, 'rolo': 75, 'sachs': 76, 'saimaza': 77, 'scrabble': 78, 'sdlg': 79, 'shua': 80, 'sixpoint': 81, 'slazenger': 82, 'steeden': 83, 'stolichnaya': 84, 'swix': 85, 'taoranju': 86, 'tapsin': 87, 'titleist': 88, 'toblerone': 89, 'univox': 90, 'vango': 91, 'vico': 92, 'weibo': 93, 'wiesmann': 94, 'winsor': 95, 'yangshengtang': 96, 'yonex': 97, 'yunnanbaiyao': 98, 'zoegas': 99}
image_set = set()
 
category_item_id = -1
image_id = 20180000000
annotation_id = 0
 
def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id
 
def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id
 
def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])
 
    annotation_item['segmentation'].append(seg)
 
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    # if category_id == 73:
    #     os.system('pause')
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)
 
def _read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file) as f:
        for line in f:
            ids.append(line.rstrip())
    return ids
 
"""通过txt文件生成"""
#split ='train' 'va' 'trainval' 'test'
def parseXmlFiles_by_txt(json_save_path):
    print("hello")
    labelfile='train_list.txt'
    image_ids = open(labelfile, encoding='utf-8').readlines()
    for image_id in image_ids:
        [jpg_set,xml_set] = image_id.split(' ')
        
        xml_file=xml_set[:-1]
        # ic(xml_file)
 
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None
 
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
 
        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
 
            if elem.tag == 'folder':
                continue
 
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
 
            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None
 
                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]
 
                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)
 
                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)
 
                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                                                                   bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)
    json.dump(coco, open(json_save_path, 'w'))
 
"""直接从xml文件夹中生成"""
def parseXmlFiles(xml_path,json_save_path):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue
 
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None
 
        xml_file = os.path.join(xml_path, f)
        print(xml_file)
 
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
 
        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
 
            if elem.tag == 'folder':
                continue
 
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
 
            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None
 
                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]
 
                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)
 
                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)
 
                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                                                                   bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)
    json.dump(coco, open(json_save_path, 'w'))
 
 
 
if __name__ == '__main__':
    # 通过txt文件生成
    #voc_data_dir="Annotations"
    #json_save_path="train.json"
    #parseXmlFiles_by_txt(json_save_path)
 
    # #通过文件夹生成
     ann_path="val_json"
     json_save_path="val.json"
     parseXmlFiles(ann_path,json_save_path)
```

```python
import os
# txt = []
# image_ids = open('class.txt', encoding='utf-8').readlines()
# for line in image_ids:
#     txt.append(line.strip())
# print(txt)

# f = open('coco_labels1.txt', 'w',encoding='utf-8')
# num = 0
coco = dict()
# coco['images'] = []
# coco['type'] = 'instances'
coco['annotations'] = []
# coco['categories'] = []
# for id in image_ids:
#     f.write('\''+id[:-1]+'\', ')
# f.close()

# path = "class.txt"
# files= os.listdir(path)
# for file in files:
#     print(file)
path = 'D:/dataset/LogoDet_3k/LogoDet-3K/images100' #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
num = 0
categories = {}
category_item = dict()
for file in files: #遍历文件夹
    print(file)
    # f.write('\''+id[:-1]+'\':'+str(num)+', ')
    
    category_item[file] = num
    # category_item['supercategory'] = 'none'
    # category_item['id'] = num
    # category_item['name'] = file
    
    num+=1

print(category_item)
# f.write('\n'+str(coco['categories']))
# f.close()
```

