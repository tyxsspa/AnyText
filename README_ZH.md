# AnyText：多语言视觉文本生成与编辑

<a href='https://arxiv.org/abs/2311.03054'><img src='https://img.shields.io/badge/论文-Arxiv-red'></a> <a href='https://github.com/tyxsspa/AnyText'><img src='https://img.shields.io/badge/代码-Github-green'></a> <a href='https://modelscope.cn/studios/damo/studio_anytext'><img src='https://img.shields.io/badge/演示-ModelScope-lightblue'></a> <a href='https://huggingface.co/spaces/modelscope/AnyText'><img src='https://img.shields.io/badge/演示-HuggingFace-yellow'></a>

![样本](docs/sample.jpg "样本")

## 📌新闻
[2024.02.06] - 新春快乐！我们在 [ModelScope](https://modelscope.cn/studios/iic/MemeMaster/summary) 和 [HuggingFace](https://huggingface.co/spaces/martinxm/MemeMaster) 上推出了一个有趣的应用程序（表情包大师/MeMeMaster），用于创建可爱的表情贴纸。来玩玩看吧！  
[2024.01.17] - 🎉AnyText 被 ICLR 2024 接受（**聚光灯报告**）！  
[2024.01.04] - FP16 推理现已可用，速度提升 3 倍！现在演示可以部署在大于 8GB 内存的 GPU 上。尽情享受吧！  
[2024.01.04] - HuggingFace 在线演示现已可用 [这里](https://huggingface.co/spaces/modelscope/AnyText)！  
[2023.12.28] - ModelScope 在线演示现已可用 [这里](https://modelscope.cn/studios/damo/studio_anytext/summary)！  
[2023.12.27] - 🧨我们发布了最新的检查点（v1.1）和推理代码，在 [ModelScope](https://modelscope.cn/models/damo/cv_anytext_text_generation_editing/summary) 上查看中文版。  
[2023.12.05] - 论文可在 [这里](https://arxiv.org/abs/2311.03054) 获取。  

想了解我们团队更多与 AIGC 相关的工作，请访问 [这里](https://github.com/AIGCDesignGroup)，我们正在寻找合作伙伴和研究实习生（[给我们发邮件](mailto:cangyu.gyf@alibaba-inc.com)）。

## ⏰待办事项
- [x] 发布模型和推理代码
- [x] 提供公开可访问的演示链接
- [ ] 提供免费字体文件（🤔）
- [ ] 发布合并来自社区模型或 LoRAs 重量的工具
- [ ] 在 stable-diffusion-webui 中支持 AnyText（🤔）
- [ ] 发布 AnyText-benchmark 数据集和评估代码
- [ ] 发布 AnyWord-3M 数据集和训练代码 

## 💡方法论
AnyText 包含一个扩散管道，主要由两个元素组成：辅助潜在模块和文本嵌入模块。前者使用文本字形、位置和掩蔽图像等输入生成文本生成或编辑的潜在特征。后者使用 OCR 模型对笔画数据进行编码，并将其嵌入与图像字幕嵌入结合，从分词器生成与背景无缝融合的文本。我们使用文本控制扩散损失和文本感知损失进行训练，以进一步提高书写精度。

![框架](docs/framework.jpg "框架")

## 🛠安装
```bash
# 安装 git（如果已完成则跳过）
conda install -c anaconda git
# 克隆 anytext 代码
git clone https://github.com/tyxsspa/AnyText.git
cd AnyText
# 准备一个字体文件；推荐使用 Arial Unicode MS，**您需要自己下载**
mv your/path/to/arialuni.ttf ./font/Arial_Unicode.ttf
# 创建新环境并按如下安装包：
conda env create -f environment.yaml
conda activate anytext
```

## 🔮推理
**[推荐]**：我们在 [ModelScope](https://modelscope.cn/studios/damo/studio_anytext/summary) 和 [HuggingFace](https://huggingface.co/spaces/modelscope/AnyText) 上发布了一个演示！

AnyText 包含两种模式：文本生成和文本编辑。运行下面的简单代码以在两种模式下执行推理，并验证环境是否正确安装。
```bash
python inference.py
```
如果您有高级 GPU（至少有 8GB 内存），建议如下部署我们的演示，其中包括使用说明、用户界面和丰富的示例。
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py
```
默认使用 FP16 推理，并且加载了一个中文到英文的翻译模型，以便直接输入中文提示（占用约 4GB GPU 内存）。可以修改默认行为，如下命令启用 FP32 推理并禁用翻译模型：
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py --use_fp32 --no_translator
```
如果使用 FP16 并且不使用翻译模型（或在 CPU 上加载它，[参见这里](https://github.com/tyxsspa/AnyText/issues/33)），生成一张单独的 512x512 图像将占用约 7.5GB 的 GPU 内存。
此外，可以使用其他字体文件（尽管结果可能不是最佳的）：
```bash
export CUDA_VISIBLE_DEVICES=0 && python demo.py --font_path your/path/to/font/file.ttf
```
![演示](docs/demo.jpg "演示")
**请注意**，在首次执行推理时，模型文件将被下载到：`~/.cache/modelscope/hub`。如果您需要修改下载目录，可以手动指定环境变量：`MODELSCOPE_CACHE`。

## 🌄画廊
![画廊](docs/gallery.png "画廊")


## 📈评估
我们使用句子准确度（Sen. ACC）和归一化编辑距离（NED）来评估生成文本的准确性，并使用 FID 指标来评估生成图像的质量。与现有方法相比，AnyText 在中英文本生成方面都具有显著优势。
![评价](docs/eval.jpg "评价")


## 引用
```
@article{tuo2023anytext,
      title={AnyText: 多语言视觉文本生成与编辑}, 
      author={Yuxiang Tuo and Wangmeng Xiang and Jun-Yan He and Yifeng Geng and Xuansong Xie},
      year={2023},
      eprint={2311.03054},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```