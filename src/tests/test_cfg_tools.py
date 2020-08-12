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
