_base_ = [
    '../_base_/models/resnet50.py',           # 模型设置
    '../_base_/datasets/imagenet_bs32.py',    # 数据设置
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '../_base_/default_runtime.py',           # 运行设置
]

model = dict(
    backbone=dict(
        frozen_stages=3,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=5,
              topk=(1, ),),
)

data_root = 'flower_dataset'
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file='',       # 我们假定使用子文件夹格式，因此需要将标注文件置空
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file='',       # 我们假定使用子文件夹格式，因此需要将标注文件置空
        data_prefix='val',
    ))
test_dataloader = val_dataloader

# 优化器超参数
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# 学习率策略
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)


