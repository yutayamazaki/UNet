from models import fpn, unet_resnet


def load_model(
    num_classes: int, architecture: str, backbone: str = 'resnet18',
    pretrained: bool = True
):
    """Load segmentation model which has specified backbone.
    Args:
        num_classes (int): Number of classes.
        architecture (str): 'unet' or 'fpn'.
        backbone (str): What ResNet model used as backbone.
    """
    architecture = architecture.lower()
    if architecture == 'unet':
        return unet_resnet.UNetResNet(num_classes, backbone, pretrained)
    elif architecture == 'fpn':
        return fpn.FPN(num_classes, backbone, pretrained)
    else:
        raise ValueError('Get invalid architecture name.')
