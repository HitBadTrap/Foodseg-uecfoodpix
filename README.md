# Foodseg-uecfoodpix


This repo holds the code for uecfoodpix dataset in [FoodSAM: Any Food Segmentation](https://arxiv.org/abs/2308.05938).


## Installation
Please follow our [installation.md](installation.md) to install.


## Results

| Method | mIou | aAcc | mAcc 
| :-: | :- | -: | :-: |  
|deeplabV3+ (baseline)| 65.61 |88.20| 77.56
FoodSAM | 66.14 |88.47 |78.01

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
