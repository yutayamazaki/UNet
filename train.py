import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import SegmentationDataset
from trainer import SegmentationTrainer
from unet import UNet


def load_text(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().split('\n')
    return text


def load_dataset():
    train_path = './VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    valid_path = './VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

    train_images = load_text(train_path)
    valid_images = load_text(valid_path)

    x_dir = './VOCdevkit/VOC2012/JPEGImages'
    y_dir = './VOCdevkit/VOC2012/SegmentationObject'
    X_train, y_train = [], []
    for i in train_images:
        X_train.append(os.path.join(x_dir, f'{i}.png'))
        y_train.append(os.path.join(y_dir, f'{i}.png'))

    X_valid, y_valid = [], []
    for i in valid_images:
        X_valid.append(os.path.join(x_dir, f'{i}.png'))
        y_valid.append(os.path.join(y_dir, f'{i}.png'))

    return X_train, X_valid, y_train, y_valid


def load_data():
    X = glob.glob('./VOCdevkit/VOC2012/JPEGImages/*')
    y = glob.glob('./VOCdevkit/VOC2012/SegmentationObject/*')

    train_path = './VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    valid_path = './VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

    train_images = load_text(train_path)
    valid_images = load_text(valid_path)

    X_train, X_valid = [], []
    for x in X:
        if os.path.splitext(os.path.basename(x))[0] in train_images:
            X_train.append(x)
        elif os.path.splitext(os.path.basename(x))[0] in valid_images:
            X_valid.append(x)

    y_train, y_valid = [], []
    for i in y:
        if os.path.splitext(os.path.basename(i))[0] in train_images:
            y_train.append(i)
        elif os.path.splitext(os.path.basename(i))[0] in valid_images:
            y_valid.append(i)

    return X_train, X_valid, y_train, y_valid


if __name__ == '__main__':
    batch_size = 2
    num_epochs = 50
    num_classes = 21
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(in_channels=3, num_classes=21)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    X_train, X_valid, y_train, y_valid = load_data()

    dtrain = SegmentationDataset(X_train, y_train, num_classes=21)
    train_loader = torch.utils.data.DataLoader(dtrain,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    dvalid = SegmentationDataset(X_valid, y_valid, num_classes=21)
    valid_loader = torch.utils.data.DataLoader(dvalid, batch_size=batch_size)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=3e-4)

    trainer = SegmentationTrainer(model, optimizer, criterion, num_classes=num_classes)

    for epoch in range(1, 1+num_epochs):
        train_loss = trainer.epoch_train(train_loader)
        valid_loss = trainer.epoch_eval(valid_loader)

        print(f'EPOCH: [{epoch}/{num_epochs}]')
        print(f'TRAIN LOSS: {train_loss:.3f}, VALID LOSS: {valid_loss:.3f}')
        # print(f'TRAIN IOU: {train_iou:.3f}, VALID IOU: {valid_iou:.3f}')
        torch.save(trainer.weights, f'epoch_{epoch}_loss_{valid_loss:.3f}.pth')