'''
Part of the implementation is borrowed and modified from TextDiffuser, publicly available at https://github.com/microsoft/unilm/blob/master/textdiffuser/inference.py
'''
import os
import json
import random
from tqdm import tqdm
import argparse
import numpy as np
from packaging import version
from termcolor import colored
from PIL import Image
from datasets import disable_caching
import torch
import torch.utils.checkpoint
from torchvision import transforms
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from util import filter_segmentation_mask
from model.text_segmenter.unet import UNet

disable_caching()
check_min_version("0.15.0.dev0")
PLACE_HOLDER = '*'


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5',  # no need to modify this
        help="Path to pretrained model or model identifier from huggingface.co/models. Please do not modify this.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text-to-image-with-template",
        choices=["text-to-image-with-template"],
        help="Three modes can be used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./textdiffuser_laion_generated/",
        help="output path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--classifier_free_scale",
        type=float,
        default=9.0,  # following stable diffusion (https://github.com/CompVis/stable-diffusion)
        help="Classifier free scale following https://arxiv.org/abs/2207.12598.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='textdiffuser-ckpt/diffusion_backbone',
        help='path of model'
    )
    parser.add_argument(
        "--character_segmenter_path",
        type=str,
        default='textdiffuser-ckpt/text_segmenter.pth',
        help="checkpoint of character-level segmenter"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        default=True,
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=20,
        help="Diffusion steps for sampling."
    )
    parser.add_argument(
        "--vis_num",
        type=int,
        default=4,
        help="Number of images to be sample. Please decrease it when encountering out of memory error."
    )
    parser.add_argument(
        "--a_prompt",
        type=str,
        default='best quality, extremely detailed',
        help="additional prompt"
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, watermark',
        help="negative prompt"
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
    print(f'{colored("[√]", "green")} Arguments are loaded.')
    return args


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def load_data(input_path):
    content = load_json(input_path)
    d = []
    count = 0
    for gt in content['data_list']:
        info = {}
        info['img_name'] = gt['img_name']
        info['caption'] = gt['caption']
        if PLACE_HOLDER in info['caption']:
            count += 1
            info['caption'] = info['caption'].replace(PLACE_HOLDER, " ")
        if 'annotations' in gt:
            polygons = []
            texts = []
            pos = []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0:
                    continue
                if annotation['valid'] is False:
                    continue
                polygons.append(annotation['polygon'])
                texts.append(annotation['text'])
                pos.append(annotation['pos'])
            info['polygons'] = [np.array(i) for i in polygons]
            info['texts'] = texts
            info['pos'] = pos
        d.append(info)
    print(f'{input_path} loaded, imgs={len(d)}')
    if count > 0:
        print(f"Found {count} image's caption contain placeholder: {PLACE_HOLDER}, change to ' '...")
    return d


def get_item(data_list, item):
    item_dict = {}
    cur_item = data_list[item]
    item_dict['img_name'] = cur_item['img_name']
    item_dict['caption'] = cur_item['caption']
    return item_dict


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 1000000)
    set_seed(seed)
    if os.path.exists(args.output_dir) is not True:
        os.makedirs(args.output_dir)
    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    ).cuda()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision).cuda()
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet", revision=None
    ).cuda()
    # load character-level segmenter
    segmenter = UNet(3, 96, True).cuda()
    segmenter = torch.nn.DataParallel(segmenter)
    segmenter.load_state_dict(torch.load(args.character_segmenter_path))
    segmenter.eval()
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    # setup schedulers
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.set_timesteps(args.sample_steps)
    sample_num = args.vis_num
    # inference loop
    data_list = load_data(args.json_path)
    for i in tqdm(range(len(data_list)), desc='generator'):
        item_dict = get_item(data_list, i)
        p = item_dict['img_name']
        img_name = item_dict['img_name'].split('.')[0] + '_3.jpg'
        if os.path.exists(os.path.join(args.output_dir, img_name)):
            continue
        input_image_path = os.path.join(args.glyph_dir, p)
        prompt = item_dict['caption']
        template_image = Image.open(input_image_path).resize((256, 256)).convert('RGB')

        noise = torch.randn((sample_num, 4, 64, 64)).to("cuda")  # (b, 4, 64, 64)
        input = noise  # (b, 4, 64, 64)
        captions = [prompt + ', ' + args.a_prompt] * sample_num
        captions_nocond = [args.n_prompt] * sample_num
        # encode text prompts
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.cuda()  # (b, 77)
        encoder_hidden_states = text_encoder(inputs)[0].cuda()  # (b, 77, 768)

        inputs_nocond = tokenizer(
            captions_nocond, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.cuda()  # (b, 77)
        encoder_hidden_states_nocond = text_encoder(inputs_nocond)[0].cuda()  # (b, 77, 768)
        if args.mode == 'text-to-image-with-template':
            to_tensor = transforms.ToTensor()
            image_tensor = to_tensor(template_image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)  # (b, 3, 256, 256)
            with torch.no_grad():
                segmentation_mask = segmenter(image_tensor)  # (b, 96, 256, 256)
            segmentation_mask = segmentation_mask.max(1)[1].squeeze(0)  # (256, 256)
            segmentation_mask = filter_segmentation_mask(segmentation_mask)  # (256, 256)
            segmentation_mask = torch.nn.functional.interpolate(segmentation_mask.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest')  # (b, 1, 256, 256)
            segmentation_mask = segmentation_mask.squeeze(1).repeat(sample_num, 1, 1).long().to('cuda')  # (b, 1, 256, 256)

            feature_mask = torch.ones(sample_num, 1, 64, 64).to('cuda')  # (b, 1, 64, 64)
            masked_image = torch.zeros(sample_num, 3, 512, 512).to('cuda')  # (b, 3, 512, 512)
            masked_feature = vae.encode(masked_image).latent_dist.sample()  # (b, 4, 64, 64)
            masked_feature = masked_feature * vae.config.scaling_factor  # (b, 4, 64, 64)

        # diffusion process
        intermediate_images = []
        for t in tqdm(scheduler.timesteps):
            with torch.no_grad():
                noise_pred_cond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states, segmentation_mask=segmentation_mask, feature_mask=feature_mask, masked_feature=masked_feature).sample  # b, 4, 64, 64
                noise_pred_uncond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond, segmentation_mask=segmentation_mask, feature_mask=feature_mask, masked_feature=masked_feature).sample  # b, 4, 64, 64
                noisy_residual = noise_pred_uncond + args.classifier_free_scale * (noise_pred_cond - noise_pred_uncond)  # b, 4, 64, 64
                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                input = prev_noisy_sample
                intermediate_images.append(prev_noisy_sample)

        # decode and visualization
        input = 1 / vae.config.scaling_factor * input
        sample_images = vae.decode(input.float(), return_dict=False)[0]  # (b, 3, 512, 512)

        # save pred_img
        pred_image_list = []
        for image in sample_images.float():
            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
            pred_image_list.append(image)

        for idx, img in enumerate(pred_image_list):
            img_name = item_dict['img_name'].split('.')[0]+f'_{idx}' + '.jpg'
            img.save(os.path.join(args.output_dir, img_name))


if __name__ == "__main__":
    main()
