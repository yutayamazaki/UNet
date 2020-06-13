from typing import List, Tuple

import torch
from torch.autograd._functions import Resize

from metrics import mean_intersection_over_union


class AbstractTrainer:

    def __init__(self, model, optimizer, criterion, num_classes):
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        self.num_classes = num_classes

    def epoch_train(self, train_loader):
        raise NotImplementedError()

    def epoch_eval(self, eval_loader):
        raise NotImplementedError()

    @property
    def weights(self):
        return self._model.state_dict()


class SegmentationTrainer(AbstractTrainer):

    def epoch_train(self, train_loader) -> Tuple[float, float]:
        self._model.train()
        epoch_loss: float = 0.
        iou_list: List[float] = []
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self._model(inputs)
            b, _, h, w = outputs.size()
            outputs = outputs.permute(0, 2, 3, 1)

            outputs = Resize.apply(outputs, (b*h*w, self.num_classes))
            targets = targets.reshape(-1)

            m_iou = mean_intersection_over_union(
                y_true=targets, y_pred=outputs, num_classes=self.num_classes
            )
            iou_list.append(m_iou)

            loss = self.criterion(outputs, targets)

            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        mean_loss: float = epoch_loss / len(train_loader)
        mean_iou: float = sum(iou_list) / len(train_loader)
        return mean_loss, mean_iou

    def epoch_eval(self, eval_loader) -> Tuple[float, float]:
        self._model.eval()
        epoch_loss: float = 0.
        iou_list: List[float] = []
        for inputs, targets in eval_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self._model(inputs)

            b, _, h, w = outputs.size()
            outputs = outputs.permute(0, 2, 3, 1)

            outputs = Resize.apply(outputs, (b*h*w, self.num_classes))
            targets = targets.reshape(-1)

            m_iou = mean_intersection_over_union(
                y_true=targets, y_pred=outputs, num_classes=self.num_classes
            )
            iou_list.append(m_iou)

            loss = self.criterion(outputs, targets)

            epoch_loss += loss.item()

        mean_loss: float = epoch_loss / len(eval_loader)
        mean_iou: float = sum(iou_list) / len(eval_loader)
        return mean_loss, mean_iou
