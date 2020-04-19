from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from tqdm import tqdm


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


class RecognitionTrainer(AbstractTrainer):

    def epoch_train(self, train_loader):
        self._model.train()
        epoch_loss = 0
        y_preds, y_trues = torch.Tensor([]).long(), torch.Tensor([]).long()

        for inputs, targets in tqdm(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()

            self.optimizer.zero_grad()
            outputs = F.softmax(self._model(inputs), dim=1)

            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            y_preds = torch.cat([y_preds, predicted.cpu()])
            y_trues = torch.cat([y_trues, targets.cpu()])

        epoch_loss /= len(train_loader)
        y_trues = y_trues.numpy()
        y_preds = y_preds.numpy()
        acc = accuracy_score(y_trues, y_preds)

        return epoch_loss, acc

    def epoch_eval(self, eval_loader):
        self._model.eval()
        epoch_loss = 0
        y_preds, y_trues = torch.Tensor([]).long(), torch.Tensor([]).long()

        for inputs, targets in tqdm(eval_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()
            outputs = F.softmax(self._model(inputs), dim=1)

            loss = self.criterion(outputs, targets)

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            y_preds = torch.cat([y_preds, predicted.cpu()])
            y_trues = torch.cat([y_trues, targets.cpu()])

        epoch_loss /= len(eval_loader)
        y_trues = y_trues.numpy()
        y_preds = y_preds.numpy()
        acc = accuracy_score(y_trues, y_preds)

        return epoch_loss, acc


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

            outputs = outputs.resize(b*h*w, self.num_classes)
            targets = targets.resize(b*h*w)

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

            outputs = outputs.resize(b*h*w, self.num_classes)
            targets = targets.resize(b*h*w)

            loss = self.criterion(outputs, targets)

            epoch_loss += loss.item()

        return epoch_loss / len(eval_loader)
