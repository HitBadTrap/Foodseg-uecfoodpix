# Foodseg-uecfoodpix


This repo holds the code for uecfoodpix dataset in [FoodSAM: Any Food Segmentation](https://arxiv.org/abs/2308.05938).


## Installation
a. Create a conda virtual environment and activate it.

```shell
conda create -n foodseg-uec python=3.8 -y
conda activate foodseg-uec
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).
Here we use PyTorch 1.10.1 and CUDA 11.3.
You may also switch to other version by specifying the version number.

```shell
conda install pytorch==1.10.1 torchvision==0.12.2 cudatoolkit=11.3 -c pytorch -c conda-forge -y
```

c. Install MMCV following the [official instructions](https://mmcv.readthedocs.io/en/latest/#installation). 
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

d. final
```
git clone https://github.com/HitBadTrap/Foodseg-uecfoodpix.git
cd Foodseg-uecfoodpix
pip install -e .  # or "python setup.py develop"
```


## Results

| Method | mIou | aAcc | mAcc | Weights
| :-: | :- | -: | :-: | :-: |
|deeplabV3+ (baseline)| 65.61 |88.20| 77.56 | [download](https://pan.baidu.com/s/19SoqvSsk5ID0r00V-uQlMg?pwd=kq4y)
FoodSAM | 66.14 |88.47 |78.01 | 

## Acknowledgements

A large part of the code is borrowed from [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Citation
If you want to cite our work, please use this:

```
@misc{lan2023foodsam,
      title={FoodSAM: Any Food Segmentation}, 
      author={Xing Lan and Jiayi Lyu and Hanyu Jiang and Kun Dong and Zehai Niu and Yi Zhang and Jian Xue},
      year={2023},
      eprint={2308.05938},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
