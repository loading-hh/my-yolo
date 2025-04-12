'''
Author: loading-hh
Date: 2025-03-24 19:41:39
LastEditTime: 2025-04-11 11:31:37
LastEditors: loading-hh
Description:
FilePath: \my-yolo\datasets.py
可以输入预定的版权声明、个性签名、空行等
'''
import os
import tqdm
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import pandas as pd


'''
description: 改变图片的大小,是按比例改变,缺的地方用rgb的三通道值(128, 128, 128)来填充.原图片大小都是一样的
             所以一次全部读出box的值,然后直接全改box就行了.
param {*} filename_jpg 图片路径名称
param {*} box 标注框的大小
param {*} w 新图像的宽
param {*} h 新图像的高
return {*}
'''
def get_random_data_graybar(filename_jpg, box, w, h):
    # ------------------------------#
    #   读取图像并转换成RGB图像
    # ------------------------------#
    image = Image.open(filename_jpg)
    # image = cvtColor(image)
    # ------------------------------#
    #   获得图像的高宽与目标高宽
    # ------------------------------#
    iw, ih = image.size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # ---------------------------------#
    #   将图像多余的部分加上灰条
    # ---------------------------------#
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))

    # box_resize = []
    # for boxx in box:
    #     boxx[0] = str(int(int(boxx[0]) * (nw / iw) + dx))
    #     boxx[1] = str(int(int(boxx[1]) * (nh / ih) + dy))
    #     boxx[2] = str(int(int(boxx[2]) * (nw / iw) + dx))
    #     boxx[3] = str(int(int(boxx[3]) * (nh / ih) + dy))
    #     box_resize.append(boxx)

    box[0] = int(int(box[0]) * (nw / iw) + dx)
    box[1] = int(int(box[1]) * (nh / ih) + dy)
    box[2] = int(int(box[2]) * (nw / iw) + dx)
    box[3] = int(int(box[3]) * (nh / ih) + dy)

    return new_image, box


'''
description: 读取xml文件中的标注框的坐标
param {*} xml_name是每个标注框的xml的文件名字
return {*}
'''
def read_xml(xml_name):
    etree = ET.parse(xml_name)
    root = etree.getroot()
    box = []
    for obj in root.iter('object'):
        xmin,ymin,xmax,ymax = (x.text for x in obj.find('bndbox'))
        box.append([xmin,ymin,xmax,ymax])
    return box


'''
description: 创建一个图片改后标注框的xml文件
param {*} xml_name 旧xml文件的路径名称
param {*} save_name 新xml文件的路径名称
param {*} box 新xml文件中的标注框坐标
param {*} resize_w 新xml文件中更改后的图片的宽的数值
param {*} resize_h 新xml文件中更改后的图片的高的数值
return {*}
'''
def write_xml(xml_name,save_name, box, resize_w, resize_h):
    etree = ET.parse(xml_name)
    root = etree.getroot()

    # 修改图片的宽度、高度
    for obj in root.iter('size'):
        obj.find('width').text = str(resize_w)
        obj.find('height').text = str(resize_h)

    # 修改box的值
    for obj, bo in zip(root.iter('object'), box):
        for index, x in enumerate(obj.find('bndbox')):
            x.text = bo[index]
    etree.write(save_name)


'''
description: 
param {*} sourceDir 源文件夹
param {*} targetDir 保存文件夹
param {*} resize_w 改变后的宽度
param {*} resize_h 改变后的高度
param {*} csv 框是csv的还是ymal的
return {*}
'''
def start(sourceDir, targetDir, resize_w, resize_h, csv=False):
    if csv is True:
        label_path = os.path.join(sourceDir, "label.csv")
        label = pd.read_csv(label_path).copy(deep=True)
        for i in range(len(label)):
            img_path = os.path.join(sourceDir, "images", label["img_name"][i])
            box = [label["xmin"][i], label["ymin"][i], label["xmax"][i], label["ymax"][i]]
            new_image, new_box = get_random_data_graybar(img_path, box, resize_w, resize_h)
            new_image.save(os.path.join(targetDir, "images", label["img_name"][i]))
            label.loc[i, "xmin"], label.loc[i, "ymin"], label.loc[i, "xmax"], label.loc[i, "ymax"] = new_box
            print(f"image {i} 转换完成")
        label.to_csv(os.path.join(targetDir, "labels.csv"))
        
        
class MyLoadDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, transforms = None):
        super().__init__()
        self.img_path = os.path.join(img_path, "images")
        self.csv_path = os.path.join(label_path, "labels.csv")
        self.csv = pd.read_csv(self.csv_path)
        self.transforms = transforms
        
        img = []
        label = []
        bounding_box = []
        for i in range(len(self.csv)):
            img_path = os.path.join(self.img_path, self.csv["img_name"][i])
            img.append(Image.open(img_path))
            label.append(torch.tensor(self.csv["label"][i]))
            bounding_box.append(torch.tensor([self.csv["xmin"][i], self.csv["ymin"][i], self.csv["xmax"][i], self.csv["ymax"][i]]))
        self.img = img
        self.label = label
        self.bounding_box = bounding_box
        
    def __getitem__(self, index):
        data = self.img[index]
        gt_box = self.bounding_box[index]
        label = self.label[index]
        if self.transforms is not None:
            data = self.transforms(data)
            
        return data, gt_box, label
            
    def __len__(self):
        return len(self.img)
        


if __name__ == "__main__":
    # sourceDir = r"C:\Users\CCU6\Desktop\banana-detection\banana-detection\bananas_val"
    # targetDir = r"./dataset/test"
    # start(sourceDir, targetDir, 416, 416, True)
    trans = transforms.Compose([
                            transforms.ToTensor(),   #transforms.ToTensor()会归一化
                            ])
    load = MyLoadDataset(r"C:\Users\CCU6\Desktop\自己的东西\yolov2改\my-yolo\dataset\test", 
                         r"C:\Users\CCU6\Desktop\自己的东西\yolov2改\my-yolo\dataset\test", 
                         trans)
    
    print(len(load))
    print(load[0])
