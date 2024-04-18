'''
AnyText: Multilingual Visual Text Generation And Editing
Paper: https://arxiv.org/abs/2311.03054
Code: https://github.com/tyxsspa/AnyText
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import sys
import os
import torch
from cldm.model import create_model

add_ocr = True  # merge OCR model
ocr_path = './ocr_weights/ppv3_rec.pth'


if len(sys.argv) == 3:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
else:
    print('Args are wrong, using default input and output path!')
    input_path = './models/v1-5-pruned.ckpt'  # sd1.5
    output_path = './models/anytext_sd15_scratch.ckpt'

assert os.path.exists(input_path), 'Input model does not exist.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models_yaml/anytext_sd15.yaml')

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

if add_ocr:
    ocr_weights = torch.load(ocr_path)
    if 'state_dict' in ocr_weights:
        ocr_weights = ocr_weights['state_dict']
    for key in ocr_weights:
        new_key = 'text_predictor.' + key
        target_dict[new_key] = ocr_weights[key]
    print('ocr weights are added!')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
