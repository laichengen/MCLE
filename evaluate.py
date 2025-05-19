import argparse
import json

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

results_file = 'metric/result.json'
pre_explanation_file = 'metric/pre_explanation.json'
gt_explanation_file = 'metric/gt_explanation.json'
ans_error_file = 'metric/error.json'


def eval(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, device_map="cuda",
                                                               torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model_path)
    if args.ckpt is not None:
        print(args.ckpt)
        model = PeftModel.from_pretrained(model, model_id=args.ckpt)

    testdata = json.load(open(args.test_path))
    res = []
    for case in tqdm(testdata):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": case['image'],
                    },
                    {"type": "text", "text": case['question']},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        res.append({
            "image_id": case['image_id'],
            "caption": output_text
        })
    open(results_file, 'w').write(json.dumps(res))


def get_pred_ans_exp(res_path, pre_exp_path):
    res = json.load(open(res_path))
    pred_ans = {}
    for case in res:
        exp = case['caption'][0]
        try:
            case['caption'] = exp.split('Because ')[1].split(' So the answer is')[0]
            # case['caption'] = case['caption'][0]
            pred_ans[case['image_id']] = exp.split('So the answer is ')[1][:-1]
        except:
            case['caption'] = exp
            # print(exp)
            pred_ans[case['image_id']] = 'null'

    open(pre_exp_path, 'w').write(json.dumps(res, indent=4))
    return pred_ans


def get_ground_truth_exp(test_path, gt_explanation_file):
    testSet = json.load(open(test_path))
    data = {
        'annotations': [],
        'images': []
    }
    idx = 0
    images = {}
    for case in testSet:
        for exp in case['response']:
            data['annotations'].append({
                'image_id': case['image_id'],
                'caption': exp.split('Because ')[1].split(' So the answer is')[0],
                'id': idx
            })
            idx += 1
        images[case['image_id']] = case['image']
    for id, img in images.items():
        data['images'].append({
            'file_name': img,
            'id': id
        })
    open(gt_explanation_file, 'w').write(json.dumps(data, indent=4))


def metric(args):
    # get answer and explanation
    pred_ans = get_pred_ans_exp(results_file, pre_explanation_file)
    get_ground_truth_exp(args.test_path, gt_explanation_file)

    # caption evaluation
    coco = COCO(gt_explanation_file)
    coco_result = coco.loadRes(pre_explanation_file)
    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    # answer evaluation
    data = json.load(open(args.test_path))
    result = json.load(open(results_file))
    result_dict = {}
    for case in result:
        result_dict[case['image_id']] = case
    error_case = []
    right = 0
    for case in data:
        if case['answer'] == pred_ans[case['image_id']]:
            right += 1
        else:
            case['predict'] = result_dict[case[('image_id')]]
            error_case.append(case)
    open(ans_error_file, 'w').write(json.dumps(error_case, indent=4))
    acc = right / len(data)
    print('Acc: ' + str(round(acc, 3)))


def parse_args():
    parse = argparse.ArgumentParser()  # 2、创建参数对象
    parse.add_argument('--ckpt', type=str, default=None)  # 3、往参数对象添加参数
    parse.add_argument('--model_path', type=str, default='checkpoints/Qwen2.5-SFT')
    parse.add_argument('--test_path', type=str, default='dataset/NLE/Demo/test.json')
    parse.add_argument('--metric', type=bool, default=False)
    args = parse.parse_args()  # 4、解析参数对象获得解析对象
    return args


if __name__ == '__main__':
    args = parse_args()
    eval(args)
    if args.metric:
        metric(args)
