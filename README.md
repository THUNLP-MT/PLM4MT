# PLM4MT

This is the code for our ACL 2022 work [MSP: Multi-Stage Prompting for Making Pre-trained Language Models Better Translators](http://arxiv.org/abs/2110.06609). The implementation is on top of the open-source NMT toolkit [THUMT](https://github.com/THUNLP-MT/THUMT).

## Contents

* [Prerequisites](#prerequisites)
* [mGPT](#mgpt)
* [Format](#format)
* [Training](#training)
* [Decoding](#decoding)
* [Postprocessing](#postprocessing)
* [License](#license)
* [Citation](#citation)

## Prerequisites

* Python >= 3.7
* tensorflow-cpu >= 2.0
* torch >= 1.7
* transformers

Please read the document of [THUMT](https://github.com/THUNLP-MT/THUMT/blob/master/docs/index.md) before using this Repository.

## mGPT
You can download the mGPT checkpoint at [this url](https://huggingface.co/THUMT/mGPT).

## Format

We use `<extra_id_0>` to separate a source and a target sentence. For the WMT14 En-De dataset, the training file contains lines with the following format:

```
<extra_id_5> Graphical artwork, corporate identity and corporate design. <extra_id_0> Grafische Gestaltung, Layout, Corporate Identity und Corporate Design.
```

Here `<extra_id_5>` is a tag to indicate the source language, which can be omitted.

For inference, the test set contains lines like:

```
<extra_id_5> Gutach: Increased safety for pedestrians <extra_id_0>
```




## Training

Using the following command to train a prompt for translation:

```[bash]
CODES=<path/to/this-repository>
CKPT=<path/to/mGPT-checkpoint>
export PYTHONPATH=$CODES:$PYTHONPATH

export USE_TF=0
export USE_TORCH=1

python $CODES/thumt/bin/trainer.py \
    --half \
    --input <path/to/traininig-file> \
    --model <model_name> \
    --ptm $CKPT \
    --parameters=device_list=[0,1,2,3,4,5,6,7],\
                 train_steps=40000,update_cycle=16,batch_size=256,\
                 save_checkpoint_steps=2000,max_length=256 \
    --hparam_set base
```

Here `model_name` has the following three options:

* `mgpt_prompt`: mGPT with Prompt tuning
* `mgpt_prefix`: mGPT with Prefix-tuning
* `mgpt_msp`: mGPT with multi-stage prompting

## Decoding

The following command decodes an input file:
```
CODES=<path/to/this-repository>
export PYTHONPATH=<path/to/this-repository>:$PYTHONPATH

python $CODES/thumt/bin/translator.py \
  --input <path/to/test-file> \
  --ptm <path/to/mgpt> \
  --output <path/to/output-file> \
  --model <model-name> \
  --half --prefix <path/to/trained-prompt> \
  --parameters=device_list=[0,1,2,3],\
               decode_alpha=0.0,\
               decode_batch_size=4,\
               prompt_length=128
```

## Postprocessing

We use `tools/punc.cpp` to replace punctuations for Chinese. Use the following command to compile the code:

```[bash]
g++ -std=c++11 -o punc tools/punc.cpp
```

Then use the following command to replace punctuations

```[bash]
cat <path/to/input-file> | ./punc | <path/to/output-file>
```

## License

Open source licensing is under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause), which allows free use for research purposes.

## Citation

```
@article{tan2021msp,
  title={{MSP}: Multi-stage prompting for making pre-trained language models better translators},
  author={Tan, Zhixing and Zhang, Xiangwen and Wang, Shuo and Liu, Yang},
  journal={arXiv preprint arXiv:2110.06609},
  year={2021}
}
```
