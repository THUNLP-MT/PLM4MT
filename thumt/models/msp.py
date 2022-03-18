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
import thumt.utils.summary as summary


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


class Prompt(modules.Module):

    def __init__(self, model, num_prompts, prompt_length, name="prompt"):
        super(Prompt, self).__init__(name=name)

        self.embed_dim = model.config.hidden_size
        self.split_size = self.embed_dim
        self.hidden_size = model.config.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scales = nn.Parameter(
            torch.ones([num_prompts]))
        self.add_name(self.scales, "scales")
        self.prompts = nn.Parameter(
            torch.empty(
            [
                num_prompts, 2 * self.num_decoder_layers,
                prompt_length, self.hidden_size
            ]))
        self.add_name(self.prompts, "prompts")
        self._model = [model]

        with torch.no_grad():
            for i in range(self.prompts.shape[0]):
                for j in range(self.prompts.shape[1]):
                    nn.init.xavier_uniform_(self.prompts[i, j])

    @property
    def model(self):
        return self._model[0]

    def forward(self, batch_size):
        key_values = [[] for _ in range(self.prompts.shape[0])]

        for i in range(self.prompts.shape[0]):
            for j in range(self.num_decoder_layers):
                scale = torch.maximum(torch.ones([]), self.scales[i])
                k = self.prompts[i, 2*j][None, :, :] * scale
                v = self.prompts[i, 2*j+1][None, :, :] * scale
                k = k.repeat([batch_size, 1, 1])
                v = v.repeat([batch_size, 1, 1])
                k = _split_heads(k, self.num_heads, self.head_dim)
                v = _split_heads(v, self.num_heads, self.head_dim)
                key_values[i].append((k, v))

        return key_values


class mGPT(modules.Module):

    def __init__(self, model, params, name="mgpt"):
        super(mGPT, self).__init__(name=name)
        self.params = params
        # Do not add gpt parameters to our module
        self._gpt_model = [model]

        params.hidden_size = model.config.hidden_size

        self.hidden_size = params.hidden_size
        self.num_decoder_layers = model.config.num_hidden_layers
        self.embed_dim = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if params.share_prompt:
            self.prompt_model = Prompt(model, 1, params.prompt_length)
        else:
            self.prompt_model = Prompt(model, 2+params.re_encoding,
                                       params.prompt_length)

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
        key_values = self.prompt_model.forward(1)

        state = {}

        for i, (k, v) in enumerate(key_values[0]):
            k = _combine_heads(k)
            v = _combine_heads(v)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

            state["enc_key_%d" % i] = k
            state["enc_value_%d" % i] = v

        for i, (k, v) in enumerate(key_values[1]):
            k = _combine_heads(k)
            v = _combine_heads(v)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

            state["rec_key_%d" % i] = k
            state["rec_value_%d" % i] = v

        for i, (k, v) in enumerate(key_values[2]):
            k = _combine_heads(k)
            v = _combine_heads(v)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

            state["dec_key_%d" % i] = k
            state["dec_value_%d" % i] = v

        return state

    def encode(self, features, state):
        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        pl = self.params.prompt_length

        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        key_values = self.prompt_model.forward(batch_size)

        # Prompt for the encoding stage
        past_key_values = key_values[0]

        outputs = self.gpt_model(input_ids, past_key_values,
                                 position_ids=position_ids,
                                 use_cache=True)

        past_key_values = []

        for (k, v) in outputs.past_key_values:
            # Activations in the encoding stage
            past_key_values.append((k[:, :, pl:, :], v[:, :, pl:, :]))

        state["enc_activations"] = tuple(past_key_values)

        state["key_values"] = key_values
        state["past_key_values"] = tuple(past_key_values)

        # Re-encoding
        for i in range(self.params.re_encoding):
            state = self.rencode(features, state, i)

        # Prepare for decoding
        pkv = state["past_key_values"]
        past_key_values = []

        for i in range(self.num_decoder_layers):
            # Prompt for the decoding stage
            key, value = key_values[-1][i]
            pk, pv = pkv[i]
            # Concat decoding prompt and re-encoded activations
            past_key_values.append((torch.cat([key, pk], axis=2),
                                    torch.cat([value, pv], axis=2)))

        state["past_key_values"] = past_key_values

        return state

    def rencode(self, features, state, idx):
        # Re-encoding
        input_ids = features["source"]
        batch_size = input_ids.shape[0]
        pl = self.params.prompt_length
        sl = input_ids.shape[1]
        src_mask = features["source_mask"]

        key_values = state["key_values"]
        src_length = torch.sum(features["source_mask"], 1).long()
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        pkv = state["past_key_values"]

        pfx_mask = torch.ones([batch_size, pl], device=src_mask.device)
        attention_mask = torch.cat([pfx_mask, src_mask, src_mask], dim=1)

        # Add prefixes
        past_key_values = []

        for i in range(self.num_decoder_layers):
            if not self.params.share_prompt:
                # Prompt for the re-encoding stage
                key, value = key_values[1+idx][i]
            else:
                key, value = key_values[0][i]

            pk, pv = pkv[i]
            past_key_values.append((torch.cat([key, pk], axis=2),
                                    torch.cat([value, pv], axis=2)))

        outputs = self.gpt_model(input_ids=input_ids,
                                 past_key_values=past_key_values,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 use_cache=True)

        past_key_values = []

        for (k, v) in outputs.past_key_values:
            # Activations in the re-encoding stage
            past_key_values.append((k[:, :, pl+sl:, :], v[:, :, pl+sl:, :]))

        state["past_key_values"] = tuple(past_key_values)

        return state

    def decode(self, features, state, mode="infer"):
        input_ids = features["target"]
        batch_size = input_ids.shape[0]
        src_mask = features["source_mask"]
        tgt_mask = features["target_mask"]
        state["target"] = input_ids

        pfx_mask = torch.ones([batch_size, self.params.prompt_length],
                              device=src_mask.device)
        attention_mask = torch.cat([pfx_mask, src_mask, tgt_mask],
                                   dim=1)

        input_shape = input_ids.shape
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        past_key_values = state["past_key_values"]

        if mode == "infer":
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        outputs = self.gpt_model(input_ids=input_ids,
                                 past_key_values=past_key_values,
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
        self.state = state

        # Translation loss
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
            share_prompt=False,
            re_encoding=1
        )

        return params

    @staticmethod
    def default_params(name=None):
        return mGPT.base_params()
