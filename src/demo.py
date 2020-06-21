import argparse
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

import models
import utils
from run_train import load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weights', type=str, help='The weights of trained model.'
    )
    parser.add_argument(
        '-i', '--image', type=str, default='demo.jpg',
        help='Specify the image to predict.'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='output.png',
        help='A path to save prediction result.'
    )
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config('./config.yml')
    cmaps: List[Tuple[str, Tuple[int]]] = utils.load_labelmap(
        path='../VOCdevkit/VOC2012/labelmap.txt'
    )
    print(cmaps)

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load UNet model and its weights.
    net: nn.Module = models.load_unet(
        backbone=cfg['backbone'], num_classes=cfg['num_classes']
    )
    net.load_state_dict(torch.load(args.weights_path, map_location=device))
    net.eval()
    net = net.to(device)

    # Load image used in prediction.
    img = Image.open(args.img_path).convert('RGB')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((cfg['img_size'], cfg['img_size'])),
        torchvision.transforms.ToTensor(),
    ])
    img_tensor: torch.Tensor = transform(img)
    c, h, w = img_tensor.size()
    inputs: torch.Tensor = img_tensor.view(1, c, h, w).to(device)

    # Make prediction.
    outputs: torch.Tensor = net(inputs).detach()
    _, c, h, w = outputs.size()
    out: torch.Tensor = outputs.view(c, h, w)

    # Save prediction result
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(img.resize((cfg['img_size'], cfg['img_size'])))
    ax1.set_title('original')

    anno_id, _ = os.path.splitext(os.path.basename(args.img_path))
    anno_path: str = os.path.join(
        '../VOCdevkit/VOC2012/SegmentationClass/', f'{anno_id}.png'
    )
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f'No such file: {anno_path}')
    anno_img = Image.open(anno_path).convert('RGB')
    ax2.imshow(anno_img.resize((cfg['img_size'], cfg['img_size'])))
    ax2.set_title('annotation')

    colored_image: np.ndarray = utils.create_segmentation_result(
        img=out.cpu().numpy(), cmaps=cmaps
    )

    ax3.imshow(colored_image)
    ax3.set_title('prediction')

    fig.savefig(args.output_path)
