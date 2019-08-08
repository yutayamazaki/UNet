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
    batch_size = 9
    num_epochs = 200
    num_classes = 21
    min_lr = 1e-4
    max_lr = 0.1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet(in_channels=3, num_classes=21)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    X_train, X_valid, y_train, y_valid = load_dataset()

    dtrain = SegmentationDataset(X_train, y_train, num_classes=21)
    train_loader = torch.utils.data.DataLoader(dtrain,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    dvalid = SegmentationDataset(X_valid, y_valid, num_classes=21)
    valid_loader = torch.utils.data.DataLoader(dvalid, batch_size=batch_size)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=max_lr,
                                momentum=0.9,
                                weight_decay=3e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=num_epochs,
                                                           eta_min=min_lr)

    trainer = SegmentationTrainer(model, optimizer, criterion, num_classes=num_classes)

    for epoch in range(1, 1+num_epochs):
        train_loss = trainer.epoch_train(train_loader)
        valid_loss = trainer.epoch_eval(valid_loader)

        scheduler.step()

        print(f'EPOCH: [{epoch}/{num_epochs}]')
        print(f'TRAIN LOSS: {train_loss:.3f}, VALID LOSS: {valid_loss:.3f}')

        path = os.path.join('weights', f'epoch{epoch}_loss{valid_loss:.3f}.pth')
        torch.save(trainer.weights, path)