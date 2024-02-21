import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
from cldm.recognizer import TextRecognizer, crop_image
from easydict import EasyDict as edict
from anytext_singleGPU import load_data, get_item
from tqdm import tqdm
import os
import torch
import Levenshtein
import numpy as np
import math
import argparse
PRINT_DEBUG = False
num_samples = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default='/home/yuxiang.tyx/projects/ControlNet/controlnet_wukong_generated',
        help='path of generated images for eval'
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json',
        help='json path for evaluation dataset'
    )
    args = parser.parse_args()
    return args


args = parse_args()
img_dir = args.img_dir
input_json = args.input_json

if 'wukong' in input_json:
    model_lang = 'ch'
    rec_char_dict_path = os.path.join('./ocr_weights', 'ppocr_keys_v1.txt')
elif 'laion' in input_json:
    rec_char_dict_path = os.path.join('./ocr_weights', 'en_dict.txt')


def get_ld(ls1, ls2):
    edit_dist = Levenshtein.distance(ls1, ls2)
    return 1 - edit_dist/(max(len(ls1), len(ls2)) + 1e-5)


def pre_process(img_list, shape):
    numpy_list = []
    img_num = len(img_list)
    assert img_num > 0
    for idx in range(0, img_num):
        # rotate
        img = img_list[idx]
        h, w = img.shape[1:]
        if h > w * 1.2:
            img = torch.transpose(img, 1, 2).flip(dims=[1])
            img_list[idx] = img
            h, w = img.shape[1:]
        # resize
        imgC, imgH, imgW = (int(i) for i in shape.strip().split(','))
        assert imgC == img.shape[0]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(imgH, resized_w),
            mode='bilinear',
            align_corners=True,
        )
        # padding
        padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32)
        padding_im[:, :, 0:resized_w] = resized_image[0]
        numpy_list += [padding_im.permute(1, 2, 0).cpu().numpy()]  # HWC ,numpy
    return numpy_list


def main():
    predictor = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
    rec_image_shape = "3, 48, 320"
    args = edict()
    args.rec_image_shape = rec_image_shape
    args.rec_char_dict_path = rec_char_dict_path
    args.rec_batch_num = 1
    args.use_fp16 = False
    text_recognizer = TextRecognizer(args, None)

    data_list = load_data(input_json)
    sen_acc = []
    edit_dist = []
    for i in tqdm(range(len(data_list)), desc='evaluate'):
        item_dict = get_item(data_list, i)
        img_name = item_dict['img_name'].split('.')[0]
        n_lines = item_dict['n_lines']
        for j in range(num_samples):
            img_path = os.path.join(img_dir, img_name+f'_{j}.jpg')
            img = cv2.imread(img_path)
            if PRINT_DEBUG:
                cv2.imwrite(f'{i}_{j}.jpg', img)
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).float()  # HWC-->CHW
            gt_texts = []
            pred_texts = []
            for k in range(n_lines):  # line
                gt_texts += [item_dict['texts'][k]]
                np_pos = (item_dict['positions'][k]*255.).astype(np.uint8)  # 0-1, hwc
                pred_text = crop_image(img, np_pos)
                pred_texts += [pred_text]
            if n_lines > 0:
                pred_texts = pre_process(pred_texts, rec_image_shape)
                preds_all = []
                for idx, pt in enumerate(pred_texts):
                    if PRINT_DEBUG:
                        cv2.imwrite(f'{i}_{j}_{idx}.jpg', pt)
                    rst = predictor(pt)
                    preds_all += [rst['text'][0]]
                for k in range(len(preds_all)):
                    pred_text = preds_all[k]
                    gt_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars)-1) for m in gt_texts[k]]
                    pred_order = [text_recognizer.char2id.get(m, len(text_recognizer.chars)-1) for m in pred_text]
                    if pred_text == gt_texts[k]:
                        sen_acc += [1]
                    else:
                        sen_acc += [0]
                    edit_dist += [get_ld(pred_order, gt_order)]
                    if PRINT_DEBUG:
                        print(f'pred/gt="{pred_text}"/"{gt_texts[k]}", ed={edit_dist[-1]:.4f}')
    print(f'Done, lines={len(sen_acc)}, sen_acc={np.array(sen_acc).mean():.4f}, edit_dist={np.array(edit_dist).mean():.4f}')


if __name__ == "__main__":
    main()
