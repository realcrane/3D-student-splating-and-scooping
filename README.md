# 3D Student Splatting and Scooping

Official Implementation of paper "3D Student Splatting and Scooping", CVPR 2025.

[Jialin Zhu](https://jialin.info)<sup>1*</sup>,
[Jiangbei Yue](https://scholar.google.com/citations?user=hWnY-fMAAAAJ&hl=en)<sup>2</sup>
[Feixiang He](https://scholar.google.com/citations?user=E12uw1sAAAAJ&hl=en)<sup>1</sup>
[He Wang](https://drhewang.com/)<sup>3†</sup>

<small><sup>1</sup>University College London, UK, <sup>2</sup>University of Leeds, UK<sup>3</sup>AI Centre, University College London, UK

<sup>*</sup> Work done while at UCL, <sup>†</sup> Corresponding author- he_wang@ucl.ac.uk

[Paper]()|[BibTeX](#bib)
---

## Declaration

This project is built on top of the [vanilla 3DGS code repository](https://github.com/graphdeco-inria/gaussian-splatting) and [3DGS-MCMC code repository](https://github.com/ubc-vision/3dgs-mcmc).

1 NVIDIA RTX 4090 GPU is required

We have tested the code with on Ubuntu system and WSL Ubuntu.

## Setup

There are many ways to build a 3DGS environment, and this is the most convenient one we have found. It is also suitable for servers without admin rights.

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

`--nu_degree`: initial value of nu (degree of freedom)

`--degree_lr`: learning rate of nu (degree of freedom)

`--cap_max`: maximum components number

`--scale_reg`: lambda value for scale regularization

`--opacity_reg`: lambda value for opacity regularization

`--C_burnin`: the value of C in SGHMC in burnin stage.

`--C`: the value of C in SGHMC after burnin stage.

`--burnin_iterations`: total iteration number for burnin stage.

### Visualization
The [vanilla 3DGS.](https://github.com/graphdeco-inria/gaussian-splatting) developed a viewing program with SIBR framework. Since we change the Gaussian distribution to Student's t distribution, the viewer cannot be directly applied with our SSS. We will write a visualization tool in the future if we have time.


## Notes
The parameters we provide can achieve SOTA results on three commonly used datasets (Mip-NeRF 360, Tanks&Temples, and Deep Blending). If you want to continue tuning parameters (which is possible as we are limited by time for parameters finetuing) or find parameters for your own dataset, please pay attention to the following.

Because SGHMC is a second-order sampler, the setting of learning rate is different from Adam (first-order). If you need to adjust the learning rate, please refer to the training code for a full understudying.

The C_burnin and C used by our SGHMC sampler are also parameters that are strongly correlated with the results. We save the total noise calculated by C_burnin and C in tensorboard for observation. In theory, the larger its mean value is (without causing numerical issue like inf, NaN), the better it is。


## <span id="bib">BibTex</span>
If you find our paper/project useful, please consider citing our paper:
```bibtex
@article{zhu2025sss,
  title={3D Student Splatting and Scooping},
  author={Zhu, jialin and Yue, Jiangbei and He, Feixiang and Wang, He},
  journal={CVPR},
  year={2025}
}
```
