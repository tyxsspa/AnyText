import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
import shutil
import numpy as np
import cv2
from PIL import Image, ImageFont
from torch.utils.data import DataLoader
from dataset_util import show_bbox_on_image
import argparse
from t3_dataset import T3DataSet
max_lines = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json',
        help="json path for evaluation dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/glyph_wukong',
        help="output path, clear the folder if exist",
    )
    parser.add_argument(
        "--img_count",
        type=int,
        default=1000,
        help="image count",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    dataset = T3DataSet(args.json_path, for_show=True, max_lines=max_lines, glyph_scale=2, mask_img_prob=1.0, caption_pos_prob=0.0)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    pbar = tqdm(total=args.img_count)
    for i, data in enumerate(train_loader):
        if i == args.img_count:
            break
        all_glyphs = []
        for k, glyphs in enumerate(data['glyphs']):
            all_glyphs += [glyphs[0].numpy().astype(np.int32)*255]
        glyph_img = cv2.resize(255.0-np.sum(all_glyphs, axis=0), (512, 512))
        cv2.imwrite(os.path.join(args.output_dir, data['img_name'][0]), glyph_img)
        pbar.update(1)
    pbar.close()
