'''
Author: loading-hh
Date: 2025-03-24 15:41:43
LastEditTime: 2025-03-26 10:49:20
LastEditors: loading-hh
Description:
FilePath: \my-yolo\train.py
可以输入预定的版权声明、个性签名、空行等
'''
import tqdm
import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

import argparse
from PIL.ImageFont import load_path
import matplotlib.pyplot as plt
from IPython import display

import config
import my_yolo


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", type=str, default="config.py", help="config.py path")
    parser.add_argument("--data", type=str, default="", help="dataset path")
    parser.add_argument("--lr", type=float, default=0.002, help="training learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=416, help="train, val image size (pixels)")
    parser.add_argument("--device", default="cuda", help="cuda device, i.e. cuda or cpu")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--cos_lr", type=bool, default=True, help="cosine LR scheduler")
    parser.add_argument("--seed", type=int, default=0, help="Global random training seed")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def train(opt):
    datas, lr, epochs, batch_size, img_size, device, optimizer, cos_lr, seed = (
        opt.data,
        opt.lr,
        opt.epochs,
        opt.batch_size,
        opt.img_size,
        opt.device,
        opt.optimizer,
        opt.cos_lr,
        opt.seed
    )


if __name__ == "__main__":
    opt = parse_opt()