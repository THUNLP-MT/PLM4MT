# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import tensorflow as tf
import operator

from thumt.utils.context import get_args


def sort_input_file(filename, reverse=True):
    with open(filename, "rb") as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.split())) for i, line in enumerate(inputs)]

    sorted_input_lens = sorted(input_lens, key=lambda x: x[1],
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (idx, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[idx])
        sorted_keys[idx] = i

    return sorted_keys, sorted_inputs


def _pad(x):
    max_len = 0
    masks = []

    for ids in x:
        max_len = max(len(ids), max_len)

    for ids in x:
        mask = []

        for _ in range(len(ids)):
            mask.append(1)

        for _ in range(max_len - len(ids)):
            ids.append(0)
            mask.append(0)

        masks.append(mask)

    return x, masks


def length_to_mask(lens, max_len):
    return torch.arange(max_len).expand(len(lens), max_len).cpu() < lens.unsqueeze(1)


def build_input_fn(filenames, mode, params):
    def train_input_fn(path, tokenizer, params):
        dataset = tf.data.TextLineDataset(path)
        dataset = dataset.repeat(None)
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())

        def py_tokenize(x):
            x = tokenizer.encode(x.numpy().decode("utf-8", errors="ignore"))
            return tf.convert_to_tensor(x, dtype=tf.int32)

        def map_func(x):
            return tf.py_function(py_tokenize, [x], tf.int32)

        dataset = dataset.map(map_func)

        dataset = dataset.map(
            lambda x: {
                "inputs": x,
                "lengths": tf.shape(x)[0],
            },
            num_parallel_calls=tf.data.AUTOTUNE)

        def bucket_boundaries(max_length, min_length=8, step=8):
            x = min_length
            boundaries = []

            while x <= max_length:
                boundaries.append(x + 1)
                x += step

            return boundaries

        batch_size = params.batch_size
        max_length = (params.max_length // 8) * 8
        min_length = params.min_length
        boundaries = bucket_boundaries(max_length)
        batch_sizes = [max(1, batch_size // (x - 1))
                       if not params.fixed_batch_size else batch_size
                       for x in boundaries] + [1]

        def element_length_func(x):
            return x["lengths"]

        def valid_size(x):
            size = element_length_func(x)
            return tf.logical_and(size >= min_length, size <= max_length)

        transformation_fn = tf.data.experimental.bucket_by_sequence_length(
            element_length_func,
            boundaries,
            batch_sizes,
            padded_shapes={
                "inputs": tf.TensorShape([None]),
                "lengths": tf.TensorShape([]),
                },
            padding_values={
                "inputs": 0,
                "lengths": 0,
                },
            pad_to_bucket_boundary=False)

        dataset = dataset.filter(valid_size)
        dataset = dataset.apply(transformation_fn)

        return dataset

    def infer_input_fn():
        sorted_key, sorted_data = sort_input_file(filenames)
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(sorted_data))
        dataset = dataset.shard(torch.distributed.get_world_size(),
                                torch.distributed.get_rank())
        args = get_args()
        tokenizer = args.tokenizer

        def py_tokenize(x):
            x = tokenizer.encode(x.numpy().decode("utf-8", errors="ignore"))[:-1]
            return tf.convert_to_tensor(x, dtype=tf.int32)

        def map_func(x):
            return tf.py_function(py_tokenize, [x], tf.int32)

        dataset = dataset.map(map_func)

        dataset = dataset.map(
            lambda x: {
                "source": x,
                "source_length": tf.shape(x)[0],
            },
            num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.padded_batch(
            params.decode_batch_size,
            padded_shapes={
                "source": tf.TensorShape([None]),
                "source_length": tf.TensorShape([])
            },
            padding_values={
                "source": 0,
                "source_length": 0
            })

        #dataset = dataset.map(
        #    lambda x: {
        #        "source": x["source"],
        #        "source_mask": tf.sequence_mask(x["source_length"],
        #                                        tf.shape(x["source"])[1],
        #                                        tf.float32),
        #    },

        return sorted_key, dataset

    if mode == "train":
        return train_input_fn
    elif mode == "infer":
        return infer_input_fn
    else:
        raise ValueError("Unknown mode %s" % mode)


def to_translation_features(features, sep_id, mode="train"):
    if mode == "train":
        inputs = features["inputs"]
        lengths = features["lengths"]

        sources = []
        targets = []

        inputs = inputs.numpy().tolist()
        lengths = lengths.numpy().tolist()

        for toks, length in zip(inputs, lengths):
            flag = 0
            source = []
            target = []

            for i in range(length):
                if toks[i] == sep_id:
                    flag = 1

                if flag == 0:
                    source.append(toks[i])
                else:
                    target.append(toks[i])

            sources.append(source)
            targets.append(target)

        sources, source_masks = _pad(sources)
        targets, target_masks = _pad(targets)
        target = torch.LongTensor(targets).cuda()
        target_mask = torch.FloatTensor(target_masks).cuda()

        features = {
            "source": torch.LongTensor(sources).cuda(),
            "source_mask": torch.FloatTensor(source_masks).cuda(),
            "target": target[:, :-1],
            "target_mask": target_mask[:, :-1]
        }

        return features, target[:, 1:]
    else:
        inputs = features["source"]
        lengths = features["source_length"]

        sources = []
        targets = []

        inputs = inputs.numpy().tolist()
        lengths = lengths.numpy().tolist()

        for toks, length in zip(inputs, lengths):
            flag = 0
            source = []
            target = []

            for i in range(length):
                if toks[i] == sep_id:
                    flag = 1

                if flag == 0:
                    source.append(toks[i])
                else:
                    target.append(toks[i])

            sources.append(source)
            targets.append(target)

        sources, source_masks = _pad(sources)
        targets, target_masks = _pad(targets)
        target = torch.LongTensor(targets).cuda()
        target_mask = torch.FloatTensor(target_masks).cuda()

        features = {
            "source": torch.tensor(sources).long().cuda(),
            "source_mask": torch.tensor(source_masks).float().cuda(),
        }

        return features