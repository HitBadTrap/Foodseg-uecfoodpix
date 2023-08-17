_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/uecfoodpix_320x320.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

data_preprocessor = data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(320, 320),
    test_cfg=dict(size_divisor=32))

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=103),
    auxiliary_head=dict(num_classes=103))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_bgra', reduce_zero_label=False),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(batch_size=12, dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='LoadAnnotations_bgra', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=8000,
        save_best='mIoU'))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator