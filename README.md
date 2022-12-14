# Convolutional Autoencoder Joint Boundary and Mask Adversarial Learning for Fundus Image Segmentation

The precise segmentation of the optic cup (OC) and the optic disc (OD) is important for glaucoma screening. In recent years, medical image segmentation based on convolutional neural networks (CNN) has achieved remarkable results. However, many traditional CNN methods do not consider the cross-domain problem, i.e. generalization on datasets of different domains. In this paper, we propose a novel unsupervised domain-adaptive segmentation architecture called CAE-BMAL. Firstly, we enhance the source domain with a convolutional autoencoder to improve the generalization ability of the model. Then, we introduce an adversarial learning-based boundary discrimination branch to reduce the impact of the complex environment during segmentation. Finally, we evaluate the proposed method on three datasets, Drishti-GS, RIM-ONE-r3, and REFUGE. The experimental evaluations outperform most state-of-the-art methods in accuracy and generalization. We further evaluate the cup-to-disk ratio performance in OD and OC segmentation, which indicates the effectiveness of glaucoma discrimination.

This is a pytorch implementation of Convolutional Autoencoder Joint Boundary and Mask Adversarial Learning for Fundus Image Segmentation. 

## Architecture

[Zhang X, Song J, Wang C and Zhou Z (2022) Convolutional autoencoder joint boundary and mask adversarial learning for fundus image segmentation. Front. Hum. Neurosci. 16:1043569.](https://doi.org/10.3389/fnhum.2022.1043569) 

<p align="center">
  <img src="https://github.com/CQRhinoZ/CAE-BMAL/blob/main/figure/framework.png">
</p>

## Installation

- Install Pytorch 0.4.1 (Note that the results reported in the paper are obtained by running the code on this Pytorch version. As raised by the issue, using higher version of Pytorch may seem to have a performance decrease on optic cup segmentation.)
- Clone this repo

```
git clone https://github.com/CQRhinoZ/CAE-BMAL
```

## Project Structure

- dataloaders: Load source domain, target domain and enhanced domain data
- figure: Framework figure of our paper
- network: The overall framework of the paper
- utils: Formula tools folder
- CAE.py: CAE module to generate enhanced domains
- custom_transforms.py: Modules used for drawing in the test phase
- mypath: File paths required by some modules
- re_train.py: main file for training(Some omitted part will be released when this work is published)
- test.py: main file for testing

## Dependency

After installing the dependency:

    pip install pyyaml
    pip install pytz
    pip install tensorboardX matplotlib pillow 
    pip install tqdm
    pip install scipy==1.1.0
    conda install -c conda-forge opencv

## Train

- Download datasets from [here](https://drive.google.com/file/d/1B7ArHRBjt2Dx29a3A6X_lGhD0vDVr3sy/view).
- Specify the data path in `./CAE.py`, `./mypath.py`and then train `./CAE.py`.
- Then specify the data path in `./re_train.py`, `./mypath.py`and then train `./re_train.py`.
- Save source domain model into folder `./logs`.

## Test
- Finally, specify the data path in `./test.py` and then test it.


## Citation
```
The code for source domain training is modified from [BEAL](https://github.com/emma-sjwang/BEAL/).
1. Wang, Shujun and Yu, Lequan and Li, Kang and Yang, Xin and Fu, Chi-Wing and Heng, Pheng-Ann, Boundary and Entropy-driven Adversarial Learning for Fundus Image Segmentation, International Conference on Medical Image Computing and Computer-Assisted Intervention, 2019, pp.102--110.
2. Zhang X, Song J, Wang C and Zhou Z (2022) Convolutional autoencoder joint boundary and mask adversarial learning for fundus image segmentation. Front. Hum. Neurosci. 16:1043569. doi: 10.3389/fnhum.2022.1043569
```

Feel free to contact us:

Xu ZHANG, Ph.D, Associate Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
