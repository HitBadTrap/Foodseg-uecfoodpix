# Foodseg-uecfoodpix

This repo implements the deeplabv3+ training for UECFoodPIX complete dataset.
And this repository implements the baseline for [FoodSAM: Any Food Segmentation](https://arxiv.org/abs/2308.05938).

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

d. Clone this repo.
```
git clone https://github.com/HitBadTrap/Foodseg-uecfoodpix.git
cd Foodseg-uecfoodpix
pip install -e .  # or "python setup.py develop"
```

## Testing
Run the following commands to evaluate the given checkpoint:
```
python tools/test.py [config] [checkpoint] --show-dir [output_dir] --show(optional)
```
You can append `--show` to generate visualization results in the `output_dir/vis_image`. 

For our testing example:
```
python tools/test.py ./configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_uecfoodpix-320x320.py [checkpoint] --show-dir [output_dir] --show(optional)
```

## Training
**1.** For single-gpu training, run the following command:
```
python tools/train.py [config]
```

**2.** For multi-gpu training, run the following commands:
```
bash tools/dist_train.sh [config] [num_gpu]
```
The default config is ./configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_uecfoodpix-320x320.py

For our training example:
```
# single-gpu training
python tools/train.py ./configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_uecfoodpix-320x320.py

# multi-gpu training
bash tools/dist_train.sh ./configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-80k_uecfoodpix-320x320.py 2
```

## Results

| Method | mIou | aAcc | mAcc | Model | Training Log
| :-: | :- | -: | :-: | :-: | :-: |
|deeplabV3+ (baseline)| 65.61 |88.20| 77.56 | [Link](https://pan.baidu.com/s/19SoqvSsk5ID0r00V-uQlMg?pwd=kq4y) | [Link](https://pan.baidu.com/s/1el12UBxf_DaPoI0AfzvC_w?pwd=v1xa)
[FoodSAM](https://github.com/jamesjg/FoodSAM) | 66.14 |88.47 |78.01 |    |    |


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
