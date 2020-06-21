import argparse
import logging.config
import os
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import yaml

import models
import utils
from datasets import SegmentationDataset
# from losses import lovasz_softmax
from trainer import SegmentationTrainer


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def dump_config(path: str, dic: Dict[str, Any]):
    with open(path, 'w') as f:
        yaml.dump(dic, f)


def load_text(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().split('\n')
    return text


def load_dataset():
    train_path = '../VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    valid_path = '../VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

    train_images = load_text(train_path)
    valid_images = load_text(valid_path)

    X_train, y_train = [], []
    for train_id in train_images:
        x_path: str = os.path.join(
            '../VOCdevkit/VOC2012/JPEGImages', f'{train_id}.jpg'
        )
        y_path: str = os.path.join(
            '../VOCdevkit/VOC2012/SegmentationClass', f'{train_id}.png'
        )
        x_exists: bool = os.path.exists(x_path)
        y_exists: bool = os.path.exists(y_path)
        if x_exists and y_exists:
            X_train.append(x_path)
            y_train.append(y_path)

    X_valid, y_valid = [], []
    for valid_id in valid_images:
        x_path: str = os.path.join(
            '../VOCdevkit/VOC2012/JPEGImages', f'{valid_id}.jpg'
        )
        y_path: str = os.path.join(
            '../VOCdevkit/VOC2012/SegmentationClass', f'{valid_id}.png'
        )
        x_exists: bool = os.path.exists(x_path)
        y_exists: bool = os.path.exists(y_path)
        if x_exists and y_exists:
            X_valid.append(x_path)
            y_valid.append(y_path)

    return X_train[:5], X_valid[:5], y_train[:5], y_valid[:5]


if __name__ == '__main__':
    utils.seed_everything()
    sns.set()

    with open('logger_conf.yaml', 'r') as f:
        log_config: Dict[str, Any] = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)

    logger = getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, default='./config.yml',
        help='Specify training config file.'
    )
    args = parser.parse_args()

    # Setup directory that saves the experiment results.
    dirname: str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    save_dir: str = os.path.join('../experiments', dirname)
    os.makedirs(save_dir, exist_ok=False)
    weights_dir: str = os.path.join(save_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=False)

    cfg: Dict[str, Any] = load_config(args.config)
    logger.info(f'Training configurations: {cfg}')

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.load_unet(
        backbone=cfg['backbone'], num_classes=cfg['num_classes']
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    X_train, X_valid, y_train, y_valid = load_dataset()

    dtrain = SegmentationDataset(
        X=X_train, y=y_train, num_classes=cfg['num_classes'],
        img_size=cfg['img_size']
    )
    train_loader = torch.utils.data.DataLoader(
        dtrain,
        batch_size=cfg['batch_size'],
        shuffle=True,
        drop_last=True
    )

    dvalid = SegmentationDataset(
        X=X_valid, y=y_valid, num_classes=cfg['num_classes'],
        img_size=cfg['img_size']
    )
    valid_loader = torch.utils.data.DataLoader(
        dvalid,
        batch_size=cfg['batch_size']
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg['max_lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['num_epochs'],
        eta_min=cfg['min_lr']
    )

    trainer = SegmentationTrainer(
        model, optimizer, criterion, cfg['num_classes']
    )
    best_loss = 10000.
    metrics: Dict[str, List[float]] = {
        'train_loss': [],
        'valid_loss': [],
        'train_iou': [],
        'valid_iou': []
    }
    for epoch in range(1, 1 + cfg['num_epochs']):
        train_loss, train_iou = trainer.epoch_train(train_loader)
        valid_loss, valid_iou = trainer.epoch_eval(valid_loader)

        metrics['train_loss'].append(train_loss)
        metrics['valid_loss'].append(valid_loss)
        metrics['train_iou'].append(train_iou)
        metrics['valid_iou'].append(valid_iou)

        if valid_loss < best_loss:
            best_loss = valid_loss
            name: str = cfg['backbone'].lower()
            path: str = os.path.join(
                weights_dir,
                f'{name}_loss{valid_loss:.5f}_epoch{str(epoch).zfill(3)}.pth'
            )
            torch.save(trainer.weights, path)

        scheduler.step()  # type: ignore

        logger.info(f'EPOCH: [{epoch}/{cfg["num_epochs"]}]')
        logger.info(
            f'TRAIN LOSS: {train_loss:.8f}, VALID LOSS: {valid_loss:.8f}'
        )
        logger.info(
            f'TRAIN mIoU: {train_iou:.8f}, VALID mIoU: {valid_iou:.8f}'
        )

    cfg_path: str = os.path.join(save_dir, 'config.yml')
    dump_config(cfg_path, cfg)

    # Plot metrics
    plt.plot(metrics['train_loss'], label='train')
    plt.plot(metrics['valid_loss'], label='valid')
    plt.title('Loss curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.clf()

    plt.plot(metrics['train_iou'], label='train')
    plt.plot(metrics['valid_iou'], label='valid')
    plt.title('mIoU curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'mIoU.png'))
