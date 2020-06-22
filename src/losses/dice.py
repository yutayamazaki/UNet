import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_one_hot(
    y: torch.Tensor, num_classes: int, dtype=torch.long
) -> torch.Tensor:
    b, h, w = y.size()
    # (B, H, W) -> (B, 1, H, W)
    y_tensor = y.view((b, 1, h, w)).type(dtype)
    zeros = torch.zeros((b, num_classes, h, w), dtype=dtype)
    return zeros.scatter(1, y_tensor, 1)  # type: ignore


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    input_soft: torch.Tensor = F.softmax(inputs, dim=1)

    target_one_hot: torch.Tensor = _to_one_hot(
        targets, num_classes=inputs.shape[1]
    )

    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return torch.mean(-dice_score + 1.)


class DiceLoss(nn.Module):

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(  # type: ignore
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        return dice_loss(inputs, targets, self.eps)


if __name__ == '__main__':
    inputs = torch.zeros((1, 2, 5, 5))
    inputs[:, 0, 0, 0] = 1
    targets = torch.zeros((1, 5, 5))
    targets[0, 0, 0] = 1
    print(targets)
    criterion = DiceLoss()
    loss = criterion(inputs, targets)
    print(loss)
