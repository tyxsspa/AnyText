#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 
python -m pytorch_fid \
    /data/vdb/yuxiang.tyx/AIGC/data/wukong_word/fid/wukong-40k \
    /data/vdc/yuxiang.tyx/AIGC/anytext_eval_imgs/controlnet_wukong_generated