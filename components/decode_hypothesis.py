# coding=utf-8

from asdl.asdl import *
from asdl.hypothesis import Hypothesis
from asdl.transition_system import *
import torch


class DecodeHypothesis(Hypothesis):
    def __init__(self, rec_embed=False, leap=32):
        super(DecodeHypothesis, self).__init__()
        self.action_infos = []
        self.code = None
        self.rec_embed = rec_embed
        if rec_embed:
            self.action_embed = None
            self.leap = leap

    def add_action_embedding(self, action_emb, t):
        storage_capacity = 0 if self.action_embed is None else self.action_embed.shape[0]
        emb_size = action_emb.shape[1]
        new_memory = torch.zeros(self.leap, emb_size)

        if self.action_embed is None:
            self.action_embed = new_memory
        else:
            if t >= storage_capacity:
                self.action_embed = torch.cat([self.action_embed, new_memory], dim=0)
        self.action_embed[t:, :] = action_emb

    def get_hist_action_embeddings(self, t):
        # embedding of the first t actions
        return self.action_embed[:t+1, :]

    def clone_and_apply_action_info(self, action_info):
        action = action_info.action

        new_hyp = self.clone_and_apply_action(action)
        new_hyp.action_infos.append(action_info)

        return new_hyp

    def copy(self):
        new_hyp = DecodeHypothesis(self.rec_embed)
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.action_infos = list(self.action_infos)
        new_hyp.score = self.score
        new_hyp._value_buffer = list(self._value_buffer)
        new_hyp.t = self.t
        new_hyp.code = self.code

        if self.rec_embed:
            new_hyp.action_embed = self.action_embed.clone()
            new_hyp.leap = self.leap

        new_hyp.update_frontier_info()

        return new_hyp
