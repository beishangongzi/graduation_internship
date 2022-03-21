import os
import shutil
from enum import Enum

dst_dir = os.path.join('Data/数据集')

label_file = os.path.join('Data/测试数据集-最终/label.txt')

with open(label_file, 'r') as f:
    labels = f.readlines()
class Types(Enum):
    _1 = "grass"
    _2 = "farm"
    _3 = "factory"
    _4 = "water"
    _5 = "forest"
    _6 = "building"
    _7 = "park"
i = 1
for label in labels:
    img = str(i)
    while len(img) < 3:
        img = '0' + img
    image_name = img + '.jpg'
    print(image_name)
    label_human = Types["_" + label.strip()].value
    print(label_human)
    img_path = os.path.join('Data/测试数据集-最终/', '图像', image_name)
    new_names = ['cp_' + image_name, "cp__" + image_name, 'cp___' + image_name, "cp____" + image_name]

    for new_name in new_names:
        new_path = os.path.join(dst_dir,label_human, new_name)
        shutil.copy(img_path, new_path)
    i+=1