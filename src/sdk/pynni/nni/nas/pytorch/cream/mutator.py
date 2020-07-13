# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
from nni.nas.pytorch.random import RandomMutator

_logger = logging.getLogger(__name__)


class CreamSupernetTrainingMutator(RandomMutator):
    """
    A random mutator with flops limit.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    flops_func : callable
        Callable that takes a candidate from `sample_search` and returns its candidate. When `flops_func`
        is None, functions related to flops will be deactivated.
    flops_lb : number
        Lower bound of flops.
    flops_ub : number
        Upper bound of flops.
    flops_bin_num : number
        Number of bins divided for the interval of flops to ensure the uniformity. Bigger number will be more
        uniform, but the sampling will be slower.
    flops_sample_timeout : int
        Maximum number of attempts to sample before giving up and use a random candidate.
    """
    def __init__(self, model, how_to_prob='even', pre_prob=(0.05,0.05,0.2,0.4,0.2,0.1), CHOICE_NUM=6, sta_num=(4,4,4,4,4)):

        super().__init__(model)
        self.how_to_prob = how_to_prob
        self.pre_prob = pre_prob
        self.CHOICE_NUM = CHOICE_NUM
        self.sta_num = sta_num

    def get_prob(self):
        if self.how_to_prob == 'even':
            return None
        elif self.how_to_prob == 'pre_prob':
            return self.pre_prob
        else:
            raise ValueError("prob method not supported")

    def sample_search(self):
        """
        Sample a candidate for training. When `flops_func` is not None, candidates will be sampled uniformly
        relative to flops.

        Returns
        -------
        dict
        """

        prob = self.get_prob()

        if prob is None:
            get_random_cand = [np.random.choice(self.CHOICE_NUM, item).tolist() for item in self.sta_num]
        else:
            get_random_cand = [np.random.choice(self.CHOICE_NUM, item, prob).tolist() for item in self.sta_num]

        get_random_cand.insert(0, [0])
        get_random_cand.append([0])

        return get_random_cand

    def sample_final(self):
        """
        Implement only to suffice the interface of Mutator.
        """
        return self.sample_search()
