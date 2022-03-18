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


def length_to_mask(lens, max_len):
    return torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)


def _split_heads(tensor, num_heads, attn_head_size):
    """
    Splits hidden_size dim into attn_head_size and num_heads
    """
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(*new_shape)
    return tensor.permute(0, 2, 1, 3)


def _combine_heads(x):
    batch = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    channels = x.shape[3]

    y = torch.transpose(x, 2, 1)

    return torch.reshape(y, [batch, length, heads * channels])


class GPT2PrefixModel(modules.Module):

    def __init__(self, model, prompt_length, hidden_size):
        super(GPT2PrefixModel, self).__init__(name="prefix")

        self._num_hidden_layers = model.config.num_hidden_layers
        self._prompt_length = prompt_length
        self._hidden_size = hidden_size
        self._emb_size = model.config.hidden_size
        self._num_heads = model.config.num_attention_heads
        self._head_dim = self._emb_size // self._num_heads

        with utils.scope("prefix"):
            self.emb = nn.Parameter(
                torch.empty([prompt_length, self._emb_size]))
            self.add_name(self.emb, "emb")
            self.mlp1 = modules.Affine(self._emb_size, hidden_size,
                                       name="mlp1")
            self.mlp2 = modules.Affine(hidden_size,
                self._emb_size * 2 * self._num_hidden_layers,
                name="mlp2")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb)
        nn.init.xavier_uniform_(self.mlp1.weight)
        nn.init.xavier_uniform_(self.mlp2.weight)

    def forward(self, batch_size):
        prefix_cat = self.mlp2(torch.tanh(self.mlp1(self.emb)))
        prefix_list = torch.reshape(
            prefix_cat,
            [-1, self._emb_size, 2 * self._num_hidden_layers])

        prefix_list = torch.unbind(prefix_list, -1)

        prefixes = []

        for i in range(self._num_hidden_layers):
            k = prefix_list[2*i]
            v = prefix_list[2*i + 1]
            k = k.unsqueeze(0).repeat([batch_size, 1, 1])
            v = v.unsqueeze(0).repeat([batch_size, 1, 1])
            k = _split_heads(k, self._num_heads, self._head_dim)
            v = _split_heads(v, self._num_heads, self._head_dim)
            prefixes.append((k, v))

        return prefixes


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

        self._network = GPT2PrefixModel(model, params.prompt_length,
                                        self.hidden_size // 2)

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)

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

    def generate_prefix(self):
        past_key_values = self._network.forward(1)

        state = {}

        for i, (k, v) in enumerate(past_key_values):
            k = _combine_heads(k)
            v = _combine_heads(v)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

            state["key_%d" % i] = k
            state["value_%d" % i] = v

        return state

    def encode(self, features, state):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        pl = self.params.prompt_length

        src_length = torch.sum(features["source_mask"], 1).long()
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        # Add prefixes
        past_key_values = self._network.forward(batch_size)

        outputs = self.gpt_model(input_ids, past_key_values,
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
            num_prompts=10,
            prompt_length=128,
            label_smoothing=0.1,
            sep_id=250099,
            dec_no_prefix=False,
        )

        return params

    @staticmethod
    def default_params(name=None):
        return mGPT.base_params()
