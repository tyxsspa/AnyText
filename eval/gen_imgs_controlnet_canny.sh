#!/bin/bash
python controlnet_multiGPUs.py \
        --model_path /home/yuxiang.tyx/projects/AnyText/models/control_sd15_canny.pth \
        --json_path /data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json \
        --glyph_dir /data/vdb/yuxiang.tyx/AIGC/data/wukong_word/glyph_wukong \
        --output_dir ./controlnet_wukong_generated \
        --gpus 0,1,2,3,4,5,6,7
