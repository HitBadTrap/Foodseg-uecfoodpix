_base_ = './deeplabv3plus_r50-d8_4xb4-80k_uecfoodpix-320x320.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))