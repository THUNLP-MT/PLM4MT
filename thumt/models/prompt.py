# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

import thumt.utils as utils
import thumt.modules as modules
import torch.distributed as dist


class mGPT(modules.Module):

    def __init__(self, model, params, name="mgpt"):
        super(mGPT, self).__init__(name=name)
        self.params = params
        self._gpt_model = [model]

        params.hidden_size = model.config.hidden_size

        self.hidden_size = params.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.embed_dim = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.prompt = nn.Parameter(
            torch.empty([params.prompt_length, self.hidden_size]))
        self.add_name(self.prompt, "prompt")

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prompt)

    @property
    def gpt_model(self):
        return self._gpt_model[0]

    @property
    def src_embedding(self):
        return self.gpt_model.get_input_embeddings().weight

    @property
    def tgt_embedding(self):
        return self.gpt_model.get_input_embeddings().weight

    @property
    def softmax_embedding(self):
        return self.tgt_embedding

    def load_prefix(self, path):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"])

    def encode(self, features, state):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        pl = self.params.prompt_length

        src_length = torch.sum(features["source_mask"], 1).long()
        position_ids = torch.arange(0, input_ids.shape[-1] + pl, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1,
            input_ids.shape[-1] + pl)

        inputs = torch.nn.functional.embedding(input_ids, self.src_embedding)
        inputs = torch.cat(
            [self.prompt.unsqueeze(0).repeat([batch_size, 1, 1]), inputs],
            axis=1)

        # Add prefixes
        # past_key_values = self._network.forward(batch_size)

        outputs = self.gpt_model(inputs_embeds=inputs,
                                 position_ids=position_ids,
                                 use_cache=True)

        if self.params.dec_no_prefix:
            past_key_values = []

            for (k, v) in outputs.past_key_values:
                past_key_values.append((k[:, :, pl:, :], v[:, :, pl:, :]))
        else:
            past_key_values = outputs.past_key_values

        state["past_key_values"] = tuple(past_key_values)
        state["source_length"] = src_length

        return state

    def decode(self, features, state, mode="infer"):
        input_ids = features["target"]
        batch_size = input_ids.shape[0]
        src_mask = features["source_mask"]
        tgt_mask = features["target_mask"]

        if self.params.dec_no_prefix:
            attention_mask = torch.cat([src_mask, tgt_mask], dim=1)
        else:
            pfx_mask = torch.ones([batch_size, self.params.prompt_length],
                                  device=src_mask.device)
            attention_mask = torch.cat([pfx_mask, src_mask, tgt_mask],
                                       dim=1)

        input_shape = input_ids.shape
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        position_ids = state["source_length"].unsqueeze(1) + position_ids
        position_ids = position_ids + self.params.prompt_length

        if mode == "infer":
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        outputs = self.gpt_model(input_ids=input_ids,
                                 past_key_values=state["past_key_values"],
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 use_cache=True)

        logits = outputs.logits

        if mode == "infer":
            logits = logits[:, 0, :]

        state["past_key_values"] = outputs.past_key_values

        return logits, state

    def forward(self, features, labels):
        mask = features["target_mask"]
        state = {}
        state = self.encode(features, state)

        logits, state = self.decode(features, state, "train")

        logits = logits.reshape([logits.shape[0] * logits.shape[1], -1])
        loss = self.criterion(logits, labels)
        loss = torch.sum(loss * mask) / torch.sum(mask)

        return loss

    def empty_state(self, batch_size, device):
        return {}

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    @staticmethod
    def base_params():
        params = utils.HParams(
            prompt_length=128,
            label_smoothing=0.1,
            sep_id=250099,
            dec_no_prefix=False,
        )

        return params

    @staticmethod
    def default_params(name=None):
        return mGPT.base_params()
