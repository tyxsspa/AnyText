#!/bin/bash
python eval/anytext_multiGPUs.py \
        --model_path models/anytext_v1.1.ckpt \
        --json_path /data/vdb/yuxiang.tyx/AIGC/data/laion_word/test1k.json \
        --output_dir ./anytext_laion_generated \
        --gpus 0,1,2,3,4,5,6,7
