"""
Implements the simple black-box attack from
https://github.com/cg563/simple-blackbox-attack/blob/master/simba_single.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from attacks.score.score_black_box_attack import ScoreBlackBoxAttack
from utils.compute import lp_step, sign


class SimpleAttack(ScoreBlackBoxAttack):
    """
    Simple Black-Box Attack
    """

    def __init__(self, max_loss_queries, epsilon, p, lb, ub, delta, batch_size, name):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(max_extra_queries=np.inf,
                         max_loss_queries=max_loss_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size = batch_size,
                         name = "SimBA")
                         
        #self.xo_t = None
        self.delta = delta
        self.perm = None
        self.best_loss = None
        self.i = 0

    def _perturb(self, xs_t, loss_fct):
        _shape = list(xs_t.shape)
        dim = np.prod(_shape[1:])
        b_sz = _shape[0]
        add_queries = 0
        if self.is_new_batch:
            #self.xo_t = xs_t.clone()
            self.i = 0
            self.perm = torch.rand(b_sz, dim).argsort(dim=1)
        if self.i == 0:
            #self.sgn_t = sign(torch.ones(_shape[0], dim))
            #fxs_t = lp_step(self.xo_t, self.sgn_t.view(_shape), self.epsilon, self.p)
            #bxs_t = self.xo_t
            loss = loss_fct(xs_t)
            self.best_loss = loss
            add_queries = 1
        diff = torch.zeros(b_sz, dim)
        # % if iterations are greater than dim
        idx = self.perm[:, self.i % dim]
        diff = diff.scatter(1, idx.unsqueeze(1), 1)
        new_xs = xs_t.clone().contiguous().view(b_sz,-1)
        # left attempt
        left_xs = lp_step(xs_t, diff.view_as(xs_t), self.delta, self.p)
        left_loss = loss_fct(left_xs)
        replace_flag = torch.tensor((left_loss > self.best_loss).astype(np.float32)).unsqueeze(1)
        #print(replace_flag.shape)
        self.best_loss = replace_flag.squeeze(1) * left_loss + (1 - replace_flag.squeeze(1)) * self.best_loss
        new_xs = replace_flag * left_xs.contiguous().view(b_sz,-1) + (1. - replace_flag) * new_xs
        # right attempt
        right_xs = lp_step(xs_t, diff.view_as(xs_t), - self.delta, self.p)
        right_loss = loss_fct(right_xs)
        # replace only those that have greater right loss and was not replaced
        # in the left attempt
        replace_flag = torch.tensor((right_loss > self.best_loss).astype(np.float32)).unsqueeze(1) * (1 - replace_flag)
        #print(replace_flag.shape)
        self.best_loss = replace_flag.squeeze(1) * right_loss + (1 - replace_flag.squeeze(1)) * self.best_loss
        new_xs = replace_flag * right_xs.contiguous().view(b_sz,-1) + (1 - replace_flag) * new_xs
        self.i += 1
        # number of queries: add_queries (if first iteration to init best_loss) + left queries + (right queries if any)
        num_queries = add_queries + torch.ones(b_sz) + torch.ones(b_sz) * replace_flag.squeeze(1)
        return new_xs.view_as(xs_t), num_queries

    def _config(self):
        return {
            "name": self.name, 
            "p": self.p,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "lb": self.lb,
            "ub": self.ub,
            "max_extra_queries": "inf" if np.isinf(self.max_extra_queries) else self.max_extra_queries,
            "max_loss_queries": "inf" if np.isinf(self.max_loss_queries) else self.max_loss_queries,
            "attack_name": self.__class__.__name__
        }
