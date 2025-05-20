# [MCLE](https://arxiv.org/abs/2312.13594)
Official Code for [Towards More Faithful Natural Language Explanation Using Multi-Level Contrastive Learning in VQA](https://arxiv.org/abs/2312.13594) 


### Requirements
- [PyTorch](https://pytorch.org/) 2.3.0
- [qwen-vl-utils](https://github.com/QwenLM/Qwen2.5-VL.git) (install with `pip install qwen-vl-utils[decord]==0.0.8`)
- [transformers](https://huggingface.co/docs/transformers/index) (install with `pip install transformers`)
- [peft](https://github.com/huggingface/peft.git) (install with `pip install peft`)

## Quick Start


### Model Download
The fine-tuned checkpoint of Qwen2.5-VL-3B-Instruct on VQA-X dataset can be downloaded here: [Qwen-SFT](https://pan.quark.cn/s/d416184b2b7f). Place it in `checkpoints/Qwen2.5-SFT`.


### Dataset 
The training set of demo can be found in `dataset/NLE/Demo/train.json`, where the `ratio > 0` denotes that the case is positive, the `ratio < 0` denotes that the case is negative.

The test set of demo can  be found in  `dataset/NLE/Demo/test.json`, where the image is:

<p align="center">
<img src="dataset/NLE/Demo/val/demo_val.jpg" width="512"/>
</p>

the question is `What kind of building is this?`.

the ground-truth is `Because it contains a large light at the top. So the answer is lighthouse.`.

### Code

First, you can evaluate the test set with the ` Qwen2.5-SFT`, which is the checkpoint of Qwen2.5-VL-3B-Instruct that fine-tuned on VQA-X:

```bash
python evaluate.py
```

The output of is `Because it has a lighthouse on top. So the answer is ship.`, which is inconsistent with the ground-truth.

Second, you can train the `Qwen2.5-SFT` model with Multi-Level Contrastive Learning.

```bash
python train.py
```

Third, you can evaluate the test set with the `MCLE`, which is the checkpoint of `Qwen2.5-SFT` that trained on Demo train set with Multi-Level Contrastive Learning:

```bash
python evaluate.py --ckpt output/MCLE/checkpoint-3
```

The output of is `Because it has a lighthouse on top. So the answer is lighthouse`, which reduces the inconsistency of multimodal LLM in VQA.

## Training on VQA-X dataset

### Model Download
We utilize `Qwen2.5-VL-3B-Instruct` that released by Alibaba as the backbone in our model. The pretrained model can be downloaded here: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). Place it in `checkpoints/Qwen2.5-VL-3B-Instruct`.

### Dataset Download
The training set of VQA-X dataset with part of negatives samples can be found in `dataset/NLE/VQA-X/train.json`.

The test set of VQA-X  can  be found in  `dataset/NLE/VQA-X/test.json`

The image files can be downloaded here: [COCO](https://cocodataset.org/#download) `train2014` and `val2014`. Place it in `dataset/NLE/VQA-X`.

### Code

Train the Qwen2.5-VL-3B-Instruct model with Multi-Level Contrastive Learning.

```bash
python train.py --model_path checkpoints/Qwen2.5-VL-3B-Instruct --train_path dataset/NLE/VQA-X/train.json --learning_rate 1e-5
```
Evaluate the model on the VQA-X test set with the checkpoint of `Qwen2.5-VL-3B-Instruct` that trained on VQA-X train set.

```bash
python evaluate.py --model_path checkpoints/Qwen2.5-VL-3B-Instruct --test_path dataset/NLE/VQA-X/test.json --ckpt output/MCLE/checkpoint-** --metric True
```
