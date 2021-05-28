# coding=utf-8

from asdl.asdl import *
from asdl.hypothesis import Hypothesis
from asdl.transition_system import *
import torch


class DecodeHypothesis(Hypothesis):
    def __init__(self, rec_embed=False):
        super(DecodeHypothesis, self).__init__()

        # new feature for transformer: record input embedding
        self.rec_embed = rec_embed
        if rec_embed:
            self.action_embed = []
        self.action_infos = []
        self.code = None

    def get_hist_action_embeddings(self):
        # (tgt_action_len, action_embed_size)
        return torch.stack(self.action_embed, dim=0)

    def clone_and_apply_action_info(self, action_info):
        action = action_info.action

        new_hyp = self.clone_and_apply_action(action)
        new_hyp.action_infos.append(action_info)

        return new_hyp

    def copy(self):
        new_hyp = DecodeHypothesis()
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.action_infos = list(self.action_infos)
        new_hyp.score = self.score
        new_hyp._value_buffer = list(self._value_buffer)
        new_hyp.t = self.t
        new_hyp.code = self.code

        new_hyp.update_frontier_info()

        return new_hyp
