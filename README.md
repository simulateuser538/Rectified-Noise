# [AAAI 2026] Rectified Noise: A Generative Model Using Positive-incentive Noise

![Visualization of the $\pi$-noise by $\Delta$RN.](assets/visual.png)

<br>
<a href="https://arxiv.org/pdf/2511.07911"><img src="https://img.shields.io/static/v1?label=Paper&message=2511.07911&color=red&logo=arxiv"></a>
<a href="https://huggingface.co/xiangzai/recitified_noise"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>

## Introduction
This is a [Pytorch](https://pytorch.org) implementation of **Rectified Noise**, a generative model using positive-incentive noise to enhance model's sampling.

![Overview of Laytrol](assets/pipeline.png)

## Setup

We provide an `environment.yml` file that can be used to create a Conda environment.

```bash
conda env create -f environment.yml
conda activate RN
```

## Usage

### Training
1. We provide a training script for RN in  `train_rectified_noise.py`

   Run:

```bash
torchrun --nnodes=1 --nproc_per_node=4  train_rectified_noise.py  \
--data-path /path/to/data \
--num-classes 3 \
--path-type Linear \
--prediction velocity  \
--ckpt /path/to/pretrained_model \
--model SiT-B/2
--learn-mu True \
--depth 1 \
```

You can find relevant checkpoint files from the previous Hugging Face link.

2. Parameters:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-path ` | str | `-` | Path to the dataset. |
| `--num-classes` | int | `-` | Number of classes. |
| `--path-type` | str | `Linear` | Directory to save the generated images. |
| `--prediction` | str | `velocity` | Output type of network. |
| `--ckpt` | str | `-` | Path to pretrained model checkpoint. |
| `--model` | str | `SiT-B/2` | Model type, any option from the model list. |
| `--learn-mu` | bool | `True` | Whether to learn the mu parameter. |
| `--depth` | int | `1` | Depth parameter for the SiTF2 model(Extra SiT Block). |

**Sampling**

1. Using the trained RN model to enhance the pre-trained model

```bash
torchrun --nnodes=1 --nproc_per_node=4  train_rectified_noise.py  \
--path-type Linear \
--prediction velocity  \
--ckpt /path/to/pretrained_model \
--sitf2-ckpt /path/to/pretrained_RN \
--model SiT-B/2
--learn-mu True \
--depth 1 \
```

## Ackownledgement
This repo benefits from [SiT](https://github.com/willisma/SiT). Thanks for their excellent works.

## Contact
If you have any question about this project, please contact mguzhenyu@outlook.com.

## Citation

If you find the code useful for your research, please consider citing our work:

```
@misc{gu2025rectifiednoisegenerativemodel,
      title={Rectified Noise: A Generative Model Using Positive-incentive Noise}, 
      author={Zhenyu Gu and Yanchen Xu and Sida Huang and Yubin Guo and Hongyuan Zhang},
      year={2025},
      eprint={2511.07911},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.07911}, 
}
```
