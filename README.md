# AnyText: Multilingual Visual Text Generation And Editing

<a href='https://arxiv.org/abs/2311.03054'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://github.com/tyxsspa/AnyText'><img src='https://img.shields.io/badge/Code-Github-green'></a> <a href='https://modelscope.cn/studios/damo/studio_anytext'><img src='https://img.shields.io/badge/Demo-ModelScope-lightblue'></a> <a href='https://huggingface.co/spaces/modelscope/AnyText'><img src='https://img.shields.io/badge/Demo-HuggingFace-yellow'></a>

![sample](docs/sample.jpg "sample")

## 📌News
[2024.01.17] - 🎉AnyText has been accepted by ICLR 2024(**Spotlight**)!  
[2024.01.04] - FP16 inference is available, 3x faster! Now the demo can be deployed on GPU with >8GB memory. Enjoy!  
[2024.01.04] - HuggingFace Online demo is available [here](https://huggingface.co/spaces/modelscope/AnyText)!  
[2023.12.28] - ModelScope Online demo is available [here](https://modelscope.cn/studios/damo/studio_anytext/summary)!  
[2023.12.27] - 🧨We released the latest checkpoint(v1.1) and inference code, check on [modelscope](https://modelscope.cn/models/damo/cv_anytext_text_generation_editing/summary) in Chinese.  
[2023.12.05] - The paper is available at [here](https://arxiv.org/abs/2311.03054).  

For more AIGC related works of our group, please visit [here](https://github.com/AIGCDesignGroup).

## ⏰TODOs
- [x] Release the model and inference code
- [x] Provide publicly accessible demo link
- [ ] Provide a free font file(🤔)
- [ ] Release tools for merging weights from community models or LoRAs
- [ ] Support AnyText in stable-diffusion-webui(🤔)
- [ ] Release AnyText-benchmark dataset and evaluation code
- [ ] Release AnyWord-3M dataset and training code
 

## 💡Methodology
AnyText comprises a diffusion pipeline with two primary elements: an auxiliary latent module and a text embedding module. The former uses inputs like text glyph, position, and masked image to generate latent features for text generation or editing. The latter employs an OCR model for encoding stroke data as embeddings, which blend with image caption embeddings from the tokenizer to generate texts that seamlessly integrate with the background. We employed text-control diffusion loss and text perceptual loss for training to further enhance writing accuracy.

![framework](docs/framework.jpg "framework")

## 🛠Installation
```bash
# Install git (skip if already done)
conda install -c anaconda git
# Clone anytext code
git clone https://github.com/tyxsspa/AnyText.git
cd AnyText
# Prepare a font file; Arial Unicode MS is recommended, **you need to download it on your own**
mv your/path/to/arialuni.ttf ./font/Arial_Unicode.ttf
# Create a new environment and install packages as follows:
conda env create -f environment.yaml
conda activate anytext
```

## 🔮Inference
**[Recommend]**： We release a demo on [ModelScope](https://modelscope.cn/studios/damo/studio_anytext/summary) and [HuggingFace](https://huggingface.co/spaces/modelscope/AnyText)!

AnyText include two modes: Text Generation and Text Editing. Running the simple code below to perform inference in both modes and verify whether the environment is correctly installed.
```bash
python inference.py
```
If you have advanced GPU (with at least 8G memory), it is recommended to deploy our demo as below, which includes usage instruction, user interface and abundant examples.
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py
```
FP16 inference is used as default, and a Chinese-to-English translation model is loaded for direct input of Chinese prompt (occupying ~4GB of GPU memory). The default behavior can be modified, as the following command enables FP32 inference and disables the translation model:
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py --use_fp32 --no_translator
```
If FP16 is used and the translation model not used(or load it on CPU, [see here](https://github.com/tyxsspa/AnyText/issues/33)), generation of one single 512x512 image will occupy ~7.5GB of GPU memory.  
In addition, other font file can be used by(although the result may not be optimal):
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py --font_path your/path/to/font/file.ttf
```
![demo](docs/demo.jpg "demo")
**Please note** that when executing inference for the first time, the model files will be downloaded to: `~/.cache/modelscope/hub`. If you need to modify the download directory, you can manually specify the environment variable: `MODELSCOPE_CACHE`.

## 🌄Gallery
![gallery](docs/gallery.png "gallery")


## 📈Evaluation
We use Sentence Accuracy (Sen. ACC) and Normalized Edit Distance (NED) to evaluate the accuracy of generated text, and use the FID metric to assess the quality of generated images. Compared to existing methods, AnyText has a significant advantage in both Chinese and English text generation.
![eval](docs/eval.jpg "eval")


## Citation
```
@article{tuo2023anytext,
      title={AnyText: Multilingual Visual Text Generation And Editing}, 
      author={Yuxiang Tuo and Wangmeng Xiang and Jun-Yan He and Yifeng Geng and Xuansong Xie},
      year={2023},
      eprint={2311.03054},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

