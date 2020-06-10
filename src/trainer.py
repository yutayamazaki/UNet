import torch
from torch.autograd._functions import Resize


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

    def epoch_train(self, train_loader):
        self._model.train()
        epoch_loss = 0.
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self._model(inputs)
            b, _, h, w = outputs.size()
            outputs = outputs.permute(0, 2, 3, 1)

            outputs = Resize.apply(outputs, (b*h*w, self.num_classes))
            targets = targets.reshape(-1)

            # Got deprecated warnings.
            # outputs = outputs.resize(b*h*w, self.num_classes)
            # targets = targets.resize(b*h*w)

            loss = self.criterion(outputs, targets)

            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return epoch_loss / len(train_loader)

    def epoch_eval(self, eval_loader):
        self._model.eval()
        epoch_loss = 0.
        for inputs, targets in eval_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self._model(inputs)

            b, _, h, w = outputs.size()
            outputs = outputs.permute(0, 2, 3, 1)

            outputs = Resize.apply(outputs, (b*h*w, self.num_classes))
            targets = targets.reshape(-1)

            # Got deprecated warnings.
            # outputs = outputs.resize(b*h*w, self.num_classes)
            # targets = targets.resize(b*h*w)

            loss = self.criterion(outputs, targets)

            epoch_loss += loss.item()

        return epoch_loss / len(eval_loader)
