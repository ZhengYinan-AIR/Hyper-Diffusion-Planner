<div align="center">

<h3>Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving</h3>

[Yinan Zheng](https://zhengyinan-air.github.io/)\*, [Tianyi Tan\*](https://github.com/0ttwhy4), Bin Huang\*, Enguang Liu, [Ruiming Liang](https://github.com/LRMbbj), Jianlin Zhang, Jianwei Cui, Guang Chen, Kun Ma, Hangjun Ye, [Long Chen](https://long.ooo/), [Ya-Qin Zhang](https://scholar.google.com/citations?user=mDOMfxIAAAAJ&hl=zh-CN&oi=ao), [Xianyuan Zhan](https://zhanzxy5.github.io/zhanxianyuan/), [Jingjing Liu](https://air.tsinghua.edu.cn/en/info/1046/1194.htm)



[**[Arxiv]**](https://arxiv.org/pdf/2602.22801) [**[Project Page]**](https://zhengyinan-air.github.io/Hyper-Diffusion-Planner/)

<img src="./assets/framework.jpeg" width=100% style="vertical-align: bottom;">
</div>

The official implementation of **Hyper Diffusion Planner**. Our work demonstrates that diffusion models, when properly designed and trained, can serve as effective and scalable E2E AD planners for complex, real-world autonomous driving tasks. **Note**: In this repository, we will release the details mentioned in the paper and provide implementations on benchmarks like NAVSIM and NuPlan for community research. Note that our design is derived from real-world vehicle experiments. Consequently, its performance on these simulated benchmarks may not fully align with its real-world efficacy, which is a known limitation of such benchmarks as discussed in our paper.
<div style="display: flex; justify-content: center; align-items: center; gap: 1%;">

  <img src="./assets/001.gif" width="32%" alt="Video 1">

  <img src="./assets/002.gif" width="32%" alt="Video 2">

  <img src="./assets/003.gif" width="32%" alt="Video 3">

</div>

<div align="center">

Real-world urban scenario testing uses model output, with only simple smoothness post-refinement.
</div>




## To Do List

The code is under cleaning and will be released gradually. Prior to this, you may refer to [Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner/) / [Flow Planner](https://github.com/DiffusionAD/Flow-Planner) for the training and inference of diffusion models and [DIPOLE](https://github.com/LRMbbj/DIPOLE) for Diffusion RL.

- [ ] Nuplan Implementation
- [ ] NAVSIM Implementation
- [x] initial repo & paper


## Bibtex

If you find our code and paper can help, please cite our paper as:

```
@article{
zheng2026unleash,
title={Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving},
author={Yinan Zheng and Tianyi Tan and Bin Huang and Enguang Liu and Ruiming Liang and Jianlin Zhang and Jianwei Cui and Guang Chen and Kun Ma and Hangjun Ye and Long Chen and Ya-Qin Zhang and Xianyuan Zhan and Jingjing Liu},
journal={arXiv preprint arXiv:2602.22801},
year={2026}
}
```