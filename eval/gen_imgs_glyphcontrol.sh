#!/bin/bash
python glyphcontrol_multiGPUs.py \
        --model_path checkpoints/laion10M_epoch_6_model_ema_only.ckpt \
        --json_path /data/vdb/yuxiang.tyx/AIGC/data/laion_word/test1k.json \
        --glyph_dir /data/vdb/yuxiang.tyx/AIGC/data/laion_word/glyph_laion \
        --output_dir ./glyphcontrol_laion_generated \
        --gpus 0,1,2,3,4,5,6,7
