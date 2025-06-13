# 3D Student Splatting and Scooping

Official Implementation of "3D Student Splatting and Scooping", an **Oral** paper in **CVPR 2025**.

[Jialin Zhu](https://jialin.info)<sup>1*</sup>,
[Jiangbei Yue](https://scholar.google.com/citations?user=hWnY-fMAAAAJ&hl=en)<sup>2</sup>
[Feixiang He](https://scholar.google.com/citations?user=E12uw1sAAAAJ&hl=en)<sup>1</sup>
[He Wang](https://drhewang.com/)<sup>3†</sup>

<small><sup>1</sup>University College London, UK, <sup>2</sup>University of Leeds, UK, <sup>3</sup>AI Centre, University College London, UK

<sup>*</sup> Work done while at UCL, <sup>†</sup> Corresponding author- he_wang@ucl.ac.uk

[CVPR Papar](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_3D_Student_Splatting_and_Scooping_CVPR_2025_paper.html)|[Arxiv Paper](https://arxiv.org/abs/2503.10148)|[BibTeX](#bib)
---

## Summary
**SSS** (Student Splatting and Scooping) is a new (unnormalized) mixture model for 3D reconstruction by improving the fundamental paradigm and formulation of [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

Core ideas in **SSS**:
1. Student's t distribution with flexible control of tail fatness for better rendering quality.
2. Negative components in (unnormalized) mixture model for introducing negative densities in 3D spatial space.
3. SGHMC (Stochastic Gradient Hamiltonian Monte Carlo) sampler for training.

## Declaration

This project is built on top of the [vanilla 3DGS code repository](https://github.com/graphdeco-inria/gaussian-splatting) and [3DGS-MCMC code repository](https://github.com/ubc-vision/3dgs-mcmc).

We have tested the code with 1 NVIDIA RTX 4090 GPU (CUDA 11.7) on Ubuntu system and WSL Ubuntu.

## Setup

This is one of the most convenient way we have found for 3DGS environment configuration. It is also suitable for machines without admin rights.

### Installation steps

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/realcrane/student-splating-scooping.git
   cd student-splating-scooping
   ```
2. **Set Up the Conda Environment:**
    ```sh
    conda create -y -n sss python=3.8
    conda activate sss
    ```
3. **Install Dependencies:**
    ```sh
    pip install plyfile tqdm torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    conda install cudatoolkit-dev=11.7 -c conda-forge
    ```
4. **Install Submodules:**
    ```sh
    pip install submodules/diff-t-rasterization submodules/simple-knn/
    ```

You can install Tensorboard on requirement.

If you encounter any issues during the setup, We recommend you to go to the [vanilla 3DGS code repository](https://github.com/graphdeco-inria/gaussian-splatting) to find a solution.

## Usage
The dataset can be downloaded from [vanilla 3DGS code repository.](https://github.com/graphdeco-inria/gaussian-splatting)
### Train
```sh
python train.py -s PATH/TO/DATA -m PATH/TO/OUTPUT --config configs/{scene}.json
```
Replace the `scene` with specific scene name. We provide config files for 7 scenes in Mip-NeRF 360 dataset, 2 scenes in Tanks&Temples dataset, and 2 scenes from Deep Blending dataset.
### Test
```sh
python render.py -m PATH/TO/OUTPUT --skip_train
```
This will render the results of test views.
### Evaluation
```sh
python metrics.py -m PATH/TO/OUTPUT
```
This will show the PSNR, SSIM, and LPIPS against ground-truth.
### Parameters
Most of the training and testing parameters can refer to [vanilla 3DGS.](https://github.com/graphdeco-inria/gaussian-splatting)

New important parameters are:

`--nu_degree`: initial value of $\nu$ (degree of freedom)

`--degree_lr`: learning rate of $\nu$ (degree of freedom)

`--cap_max`: maximum components number

`--scale_reg`: lambda value for scale regularization

`--opacity_reg`: lambda value for opacity regularization

`--C_burnin`: the value of C in SGHMC in burnin stage.

`--C`: the value of C in SGHMC after burnin stage.

`--burnin_iterations`: total iteration number for burnin stage.

### Visualization
The [vanilla 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) developed a viewing program with SIBR framework. Since we change the Gaussian distribution to Student's t distribution, the viewer cannot be directly applied with our SSS. We will write a visualization tool in the future if we have time.


## Notes
The parameters we provide can achieve SOTA results on three commonly used datasets (Mip-NeRF 360, Tanks&Temples, and Deep Blending). If you want to continue tuning parameters (which is possible as we are limited by time for parameters finetuing) or find parameters for your own dataset, please pay attention to the following context.

Because SGHMC is a second-order sampler, the setting of learning rate is different from Adam (first-order). The square of the initial learning rate is the actual learning rate. If you need to adjust the learning rate, please refer to the training code for a full understanding.

The C_burnin and C used by our SGHMC sampler are also parameters that are strongly correlated with the results. We save the total noise calculated by C_burnin and C in Tensorboard for observation. In theory, the larger its mean value is (without causing numerical issue like inf, NaN), the better it is. However, be aware that a large value may cause numerical issues (inf, NaN), so the right value needs to be found for different scenarios by attempts.


## <span id="bib">BibTex</span>
If you find our paper/project useful, please consider citing our paper:
```bibtex
@inproceedings{zhu20253d,
  title={3D Student Splatting and Scooping},
  author={Zhu, Jialin and Yue, Jiangbei and He, Feixiang and Wang, He},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  pages={21045--21054},
  year={2025},
  month={June}
}
```
