import unittest
from typing import Any, Dict

import torch
import torch.nn as nn

import cfg_tools
import losses


class LoadOptimizerTests(unittest.TestCase):

    def test_load_sgd(self):
        net: nn.Module = nn.Linear(3, 2)
        name: str = 'SGD'
        kwargs: Dict[str, Any] = {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0003
        }
        opt = cfg_tools.load_optimizer(
            params=net.parameters(),
            name=name,
            **kwargs
        )
        self.assertIsInstance(opt, torch.optim.SGD)

    def test_raise(self):
        net: nn.Module = nn.Sequential(
            torch.nn.Linear(3, 2)
        )
        name: str = 'InvalidOptimizer'
        kwargs = {}
        with self.assertRaises(ValueError):
            cfg_tools.load_optimizer(
                params=net.parameters(),
                name=name,
                **kwargs
            )


class LoadSchedulerTests(unittest.TestCase):

    def setUp(self):
        net: nn.Module = nn.Linear(3, 2)
        name: str = 'SGD'
        kwargs: Dict[str, Any] = {
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0003
        }
        self.optimizer = cfg_tools.load_optimizer(
            params=net.parameters(),
            name=name,
            **kwargs
        )

    def test_load_cosine_annealing_lr(self):
        scheduler = cfg_tools.load_scheduler(
            optimizer=self.optimizer,
            name='CosineAnnealingLR',
            **{'T_max': 10}
        )
        self.assertIsInstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingLR
        )

    def test_load_cosine_annealing_warm_restart(self):
        scheduler = cfg_tools.load_scheduler(
            optimizer=self.optimizer,
            name='CosineAnnealingWarmRestarts',
            **{'T_0': 10}
        )
        self.assertIsInstance(
            scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        )

    def test_raise(self):
        with self.assertRaises(ValueError):
            cfg_tools.load_scheduler(
                optimizer=self.optimizer,
                name='InvalidScheduler',
                **{}
            )


class LoadLossTests(unittest.TestCase):

    def test_cross_entropy(self):
        name: str = 'CrossEntropyLoss'
        params: dict = {}
        criterion = cfg_tools.load_loss(name, **params)
        self.assertIsInstance(criterion, nn.CrossEntropyLoss)

    def test_focal_loss(self):
        name: str = 'FocalLoss'
        params: dict = {}
        criterion = cfg_tools.load_loss(name, **params)
        self.assertIsInstance(criterion, losses.FocalLoss)

    def test_combo_loss(self):
        name: str = 'ComboLoss'
        params: dict = {'num_classes': 5}
        criterion = cfg_tools.load_loss(name, **params)
        self.assertIsInstance(criterion, losses.ComboLoss)

    def test_jaccard_loss(self):
        name: str = 'JaccardLoss'
        params: dict = {'num_classes': 5}
        criterion = cfg_tools.load_loss(name, **params)
        self.assertIsInstance(criterion, losses.JaccardLoss)

    def test_raise(self):
        name: str = 'InvalidLossName'
        params: dict = {}
        with self.assertRaises(ValueError):
            cfg_tools.load_loss(name, **params)
