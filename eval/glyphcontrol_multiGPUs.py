import os
import shutil
import copy
import argparse
import pathlib
import json


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.json': load_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def save(data, file_path):
    file_path = pathlib.Path(file_path)
    func_dict = {'.json': save_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](data, file_path)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default='checkpoints/laion10M_epoch_6_model_ema_only.ckpt',
        help='path to checkpoint of model'
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default='0,1,2,3,4,5,6,7',
        help='gpus for inference'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./glyphcontrol_laion_generated/',
        help="output path"
    )
    parser.add_argument(
        "--glyph_dir",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/data/laion_word/glyph_laion',
        help="path of glyph images from anytext evaluation dataset"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/data/laion_word/test1k.json',
        help="json path for evaluation dataset"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir

    tmp_dir = './tmp_dir'
    exec_path = './glyphcontrol_singleGPU.py'
    continue_gen = True  # if True, not clear output_dir, and generate rest images.

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    if not continue_gen:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    os.system('sleep 1')
    gpu_ids = [int(i) for i in args.gpus.split(',')]
    nproc = len(gpu_ids)
    all_lines = load(args.json_path)
    split_file = []
    length = len(all_lines['data_list']) // nproc
    cmds = []
    for i in range(nproc):
        start, end = i*length, (i+1)*length
        if i == nproc - 1:
            end = len(all_lines['data_list'])
        temp_lines = copy.deepcopy(all_lines)
        temp_lines['data_list'] = temp_lines['data_list'][start:end]
        tmp_file = os.path.join(tmp_dir, f'tmp_list_{i}.json')
        save(temp_lines, tmp_file)
        os.system('sleep 1')
        cmds += [f'export CUDA_VISIBLE_DEVICES={gpu_ids[i]}  && python {exec_path}  --json_path {tmp_file}  --output_dir {output_dir} --model_path {args.model_path}  --glyph_dir {args.glyph_dir} && echo proc-{i} done!']
    cmds = ' & '.join(cmds)
    os.system(cmds)
    print('Done.')
    os.system('sleep 2')
    shutil.rmtree(tmp_dir)


'''
command to kill the task after running:
$ps -ef | grep singleGPU | awk '{ print $2 }' | xargs kill -9  &&  ps -ef | grep multiproce | awk '{ print $2 }' | xargs kill -9
'''
