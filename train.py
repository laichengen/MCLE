import argparse
import json
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
import pandas as pd
from qwen_vl_utils import process_vision_info

from transformers import TrainingArguments, Trainer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(image, question, response, tokenizer, max_len):
    assistant_token = 77091
    visual_ques_exp_ans = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": response},
            ],
        }
    ]

    # 得到训练数据的inputs
    text = tokenizer.apply_chat_template(
        visual_ques_exp_ans, tokenize=False
    )
    image, _ = process_vision_info(visual_ques_exp_ans)
    inputs = tokenizer(
        text=[text],
        images=image,
        padding=True,
        return_tensors="pt",
    )

    labels = inputs.input_ids.clone()
    assistant_idx = (labels == assistant_token).nonzero().squeeze()
    labels[assistant_idx[0]][:assistant_idx[1] + 2] = IGNORE_TOKEN_ID
    inputs['labels'] = labels

    return inputs


class VLDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer, max_len):
        super(VLDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        case = self.raw_data[i]
        ret = preprocess(case['image'], case['ques'], case['response'], self.tokenizer, self.max_len)
        ret = {
            'input_ids': ret.input_ids[0],
            'attention_mask': ret.attention_mask[0],
            'pixel_values': ret.pixel_values,
            'image_grid_thw': ret.image_grid_thw[0],
            'labels': ret.labels[0],
            'ratio': case['ratio']
        }
        return ret


def load_dataset(path, tokenizer, max_len):
    train_json = json.load(open(path, "r"))
    train_dataset = VLDataset(train_json, tokenizer=tokenizer, max_len=max_len)
    return train_dataset


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_path', type=str, default='checkpoints/Qwen2.5-SFT')
    parse.add_argument('--train_path', type=str, default='dataset/NLE/Demo/train.json')
    parse.add_argument('--num_train_epochs', type=int, default=3)
    parse.add_argument('--learning_rate', type=float, default=1e-4)
    args = parse.parse_args()
    return args


class MCLETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_grid_thw=inputs['image_grid_thw'],
            labels=inputs['labels']
        )
        ratio = inputs.pop('ratio')
        loss = ratio * outputs.loss
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    args = parse_args()
    token_max_len = 512

    # load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, device_map="cuda",
                                                               torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model_path)

    # quantization model
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # load data
    train_dataloader = load_dataset(args.train_path, processor, token_max_len)
    args = TrainingArguments(
        output_dir="output/MCLE",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        save_steps=1000,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"]
    )
    trainer = MCLETrainer(
        model=model,
        args=args,
        train_dataset=train_dataloader
    )
    trainer.train()
