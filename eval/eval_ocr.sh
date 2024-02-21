#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python eval/eval_dgocr.py \
        --img_dir /data/vdc/yuxiang.tyx/AIGC/anytext_eval_imgs/controlnet_wukong_generated \
        --input_json /data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json