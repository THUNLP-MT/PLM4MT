# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import logging
import os
import re
import six
import socket
import time
import torch
import transformers

import thumt.data as data
import torch.distributed as dist
import thumt.models as models
import thumt.utils as utils
from thumt.utils.context import args_scope


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output file")
    parser.add_argument("--ptm", type=str, required=True,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--prefix", type=str, required=True,
                        help="Path to prefix parameters")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision for decoding")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        device_list=[0],
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_ratio=1.0,
        decode_length=50,
        decode_batch_size=16,
    )

    return params


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(m_name):
        return params

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_params(params, args):
    params.parse(args.parameters.lower())

    return params


def convert_to_string(tensor, tokenizer):
    ids = tensor.tolist()

    s = tokenizer.decode(ids)

    idx = s.find("</s>")

    if idx != -1:
        s = s[:idx]

    idx = s.find("<pad>")

    if idx != -1:
        s = s[:idx]

    s = s.encode("utf-8")

    return s


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def main(args):
    model_cls = models.get_model(args.model)
    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = override_params(params, args)

    dist.init_process_group("nccl", init_method=args.url,
                            rank=args.local_rank,
                            world_size=len(params.device_list))
    torch.cuda.set_device(params.device_list[args.local_rank])
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.half:
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # Create model
    with torch.no_grad():
        # Load configs
        gpt_model = transformers.GPT2LMHeadModel.from_pretrained(args.ptm)
        tokenizer = transformers.MT5Tokenizer.from_pretrained(args.ptm)
        model = model_cls(gpt_model, params)

        params.bos_id = params.sep_id
        params.eos_id = tokenizer.eos_token_id
        params.pad_id = tokenizer.pad_token_id or params.eos_id

        model.load_prefix(args.prefix)

        model = model.cuda()

        if args.half:
            model = model.half()

        gpt_model.eval()
        model.eval()

        with args_scope(tokenizer=tokenizer):
            input_fn = data.build_input_fn(args.input, "infer", params)
            sorted_key, dataset = input_fn()

        iterator = iter(dataset)
        counter = 0
        pad_max = 1024
        top_beams = params.top_beams
        decode_batch_size = params.decode_batch_size

        # Buffers for synchronization
        size = torch.zeros([dist.get_world_size()]).long()
        t_list = [torch.empty([decode_batch_size, top_beams, pad_max]).long()
                  for _ in range(dist.get_world_size())]

        all_outputs = []

        while True:
            try:
                features = next(iterator)
                features = data.to_translation_features(features, params.sep_id,
                                                        "infer")
                batch_size = features["source"].shape[0]
            except Exception as e:
                features = {
                    "source": torch.ones([1, 1]).long() * 250098,
                    "source_mask": torch.ones([1, 1]).float()
                }

                batch_size = 0

            t = time.time()
            counter += 1

            # Decode
            seqs, _ = utils.beam_search([model], features, params)

            # Padding
            pad_batch = decode_batch_size - seqs.shape[0]
            pad_beams = top_beams - seqs.shape[1]
            pad_length = pad_max - seqs.shape[2]
            seqs = torch.nn.functional.pad(
                seqs, (0, pad_length, 0, pad_beams, 0, pad_batch))

            # Synchronization
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(size)
            dist.all_gather(t_list, seqs)

            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue

            for i in range(decode_batch_size):
                for j in range(dist.get_world_size()):
                    beam_seqs = []
                    pad_flag = i >= size[j]

                    for k in range(top_beams):
                        seq = convert_to_string(t_list[j][i][k], tokenizer)

                        if pad_flag:
                            continue

                        beam_seqs.append(seq)

                    if pad_flag:
                        continue

                    all_outputs.append(beam_seqs)

            t = time.time() - t
            print("Finished batch: %d (%.3f sec)" % (counter, t))

        if dist.get_rank() == 0:
            restored_outputs = []

            if sorted_key is not None:
                for idx in range(len(all_outputs)):
                    restored_outputs.append(all_outputs[sorted_key[idx]])
            else:
                restored_outputs = all_outputs

            with open(args.output, "wb") as fd:
                if top_beams == 1:
                    for seqs in restored_outputs:
                        fd.write(seqs[0] + b"\n")
                else:
                    for idx, seqs in enumerate(restored_outputs):
                        for k, seq in enumerate(seqs):
                            fd.write(b"%d\t%d\t" % (idx, k))
                            fd.write(seq + b"\n")


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    # Pick a free port
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
        parsed_args.url = url

    world_size = infer_gpu_num(parsed_args.parameters)

    if world_size > 1:
        torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                    nprocs=world_size)
    else:
        process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
