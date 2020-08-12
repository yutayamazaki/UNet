import unittest
from typing import Any, Dict

import torch
import torch.nn as nn

import cfg_tools


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
