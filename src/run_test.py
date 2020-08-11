import argparse
import logging.config
from logging import getLogger
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

import metrics
import models
import utils
from datasets import SegmentationDataset
from run_train import load_dataset


if __name__ == '__main__':
    utils.seed_everything()

    with open('logger_conf.yaml', 'r') as f:
        log_config: Dict[str, Any] = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)

    logger = getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, default='./config.yml',
        help='configファイルを指定'
    )
    parser.add_argument(
        '-w', '--weights_path', type=str,
        default='../weights/unetresnet34_loss0.004_epoch198.pth',
        help='使用するモデルの重みのパスを指定'
    )
    args = parser.parse_args()

    cfg_dict: Dict[str, Any] = utils.load_yaml(args.config)
    cfg: utils.DotDict = utils.DotDict(cfg_dict)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.utils.load_model(
        num_classes=cfg.num_classes,
        architecture=cfg.model.architecture,
        backbone=cfg.model.backbone,
        pretrained=True
    )
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()
    model = model.to(device)

    logger.info(f'Configurations: {cfg}')

    criterion = nn.CrossEntropyLoss()

    _, X_test, _, y_test = load_dataset()

    dtest = SegmentationDataset(
        X=X_test, y=y_test, num_classes=cfg.num_classes,
        img_size=cfg.img_size
    )
    test_loader = torch.utils.data.DataLoader(
        dtest,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False
    )

    model.eval()
    target_list: List[torch.Tensor] = []
    output_list: List[torch.Tensor] = []
    for inputs, targets in tqdm(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        target_list.append(targets.detach().cpu())
        output_list.append(outputs.detach().cpu())

    outputs = torch.cat(output_list, dim=0)
    targets = torch.cat(target_list, dim=0).squeeze(1)

    loss = criterion(outputs, targets)
    iou = metrics.intersection_over_union(
        y_true=targets, y_pred=outputs, num_classes=cfg.num_classes
    )
    dice_coef: float = metrics.dice_coefficient(outputs, targets)

    # cmaps: List[Tuple[str, Tuple[int]]] = \
    #     utils.load_labelmap('../VOCDataset/labelmap.txt')

    logger.info(f'Finish testing on {len(X_test)} images.')
    logger.info(f'Loss: {loss}')
    logger.info(f'Dice coefficient: {dice_coef}')
    logger.info(f'mIoU: {np.mean(iou)}')
    logger.info('IoU:')
    # for idx, (class_name, _) in enumerate(cmaps):
    #     logger.info(f'{class_name.rjust(16)}: {iou[idx]}')
