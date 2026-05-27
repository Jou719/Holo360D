<h1 align="center">Holo360D: A Large-Scale Real-World Dataset with Continuous Trajectories for Advancing Panoramic 3D Reconstruction and Beyond</h1>

<div align="center">
  <a href="https://arxiv.org/abs/2604.22482"><img src="https://img.shields.io/badge/Project%20Page-To%20be%20released-5745BB?logo=google-chrome&logoColor=white"></a> &ensp;
  <a href="https://arxiv.org/pdf/2604.22482"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://github.com/"><img src="https://img.shields.io/static/v1?label=Code&message=GitHub&color=blue&logo=github"></a> &ensp;
  <a href="https://huggingface.co/"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow&logo=huggingface"></a> &ensp;
  <a href="https://www.modelscope.cn/"><img src="https://img.shields.io/static/v1?label=Dataset&message=ModelScope&color=purple"></a>
</div>

<div align="center">
  <img src="assets/teaser.jpg" width="1100px" alt="Teaser Image">
</div>

## 🎉 NEWS
- [2026.05.xx] 🔥 We have released a sample subset of the **Holo360D** dataset on Hugging Face, featuring 5 indoor scenes and 1 outdoor scene.

---

## ✨ Overview

While feed-forward 3D reconstruction models have advanced rapidly, they still suffer from notable performance degradation on panoramic inputs due to spherical distortions. Existing panoramic datasets are also mostly captured at discrete camera positions, which limits support for continuous multi-view trajectory learning.

**Holo360D** is introduced to address these limitations. According to the paper, it contains **10K panoramas** with aligned geometry annotations, and is designed to support panoramic 3D reconstruction research with continuous trajectories in real-world scenes.

Key characteristics (from the paper):
- Large-scale real-world 360 panorama dataset.
- Continuous trajectory capture for multi-view settings.
- Aligned geometric supervision for panoramic 3D learning.
- A benchmark setup for model fine-tuning and evaluation.

## 📦 Dataset Structure

Below is the current on-disk structure under:
`/hpc2hdd/home/jluo223/oujing/datasets/Holo360D/`

```text
Holo360D/
├── Indoor_scenes/
│   ├── Indoor_001/
│   │   ├── rgb/                # panoramic RGB images (.jpg)
│   │   ├── depth/              # depth maps (.exr)
│   │   ├── mask/               # masks (.jpg)
│   │   ├── rgb_mask/           # RGB-masked panoramas (.jpg)
│   │   └── poses/              # camera poses (.txt)
│   ├── Indoor_002/
│   ├── ...
│   └── Indoor_056/
└── Outdoor_scenes/
    ├── Outdoor_001/
    │   ├── rgb/                # panoramic RGB images (.jpg)
    │   ├── depth/
    │   │   ├── mesh_depth/             # depth maps (.exr)
    │   │   ├── pointcloud_depth/       # depth maps (.exr)
    │   │   ├── visual_mesh_depth/      # visualization (.jpg)
    │   │   └── visual_pointcloud_depth/# visualization (.jpg)
    │   ├── mask/               # masks (.jpg)
    │   ├── rgb_mask/           # RGB-masked panoramas (.jpg)
    │   └── poses/              # camera poses (.txt)
    ├── Outdoor_002/
    ├── ...
    └── Outdoor_019/
```

Notes:
- Timestamp-like file names are shared across modalities to support frame-level alignment.

## 💡 Dataset Download

Detailed download links and full-package release plan are **to be released**.

- Hugging Face: to be released
- ModelScope: to be released
- Full dataset package: to be released
- Checksum / integrity files: to be released

## 🚀 Quick Start

Loading scripts and official preprocessing/evaluation pipeline are **to be released**.

A minimal usage example (placeholder) will be provided in future updates.

## 📊 Benchmark

Benchmark details, protocols, and baseline checkpoints are **to be released**.

For methodology and current experimental results, please refer to the paper:
- https://arxiv.org/abs/2604.22482

## 📄 License

License terms are **to be released**.

## 📬 Contact

Project contact and issue template are **to be released**.

## Citation
If you find this dataset useful, please cite our paper.

```bibtex
@article{ou2026holo360d,
  title={Holo360D: A Large-Scale Real-World Dataset with Continuous Trajectories for Advancing Panoramic 3D Reconstruction and Beyond},
  author={Ou, Jing and Cao, Zidong and Ren, Yinrui and Li, Zhuoxiao and Zhu, Jinjing and Hua, Tongyan and Zhang, Shuai and Xiong, Hui and Zhao, Wufan},
  journal={arXiv preprint arXiv:2604.22482},
  year={2026}
}
```
