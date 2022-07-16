model = dict(
    # type='PAN_PP',
    type='PAN_PP_V2',
    backbone=dict(
        # type='resnet18',
        type='resnet18_csp',
        pretrained=True
    ),
    neck=dict(
        # type='FPEM_v2',
        type='FPN_v3_3',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        # type='PAN_PP_DetHead',
        type='PAN_PP_DetHead_v2',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_emb=dict(
            type='EmbLoss_v2',
            feature_dim=4,
            loss_weight=0.25
        ),
        use_coordconv=False
    )
)
data = dict(
    batch_size=8,
    num_workers=8,
    train=dict(
        type='PAN_PP_CTW',
        split='train',
        is_transform=True,
        img_size=640,
        short_size=640,
        kernel_scale=0.5, #0.7
        read_type='pil',
    ),
    test=dict(
        type='PAN_PP_CTW',
        split='test',
        short_size=640,
        read_type='pil',
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=600,
    optimizer='Adam',
    use_ex=False,
    pretrain='checkpoints/pan_pp_synth_exp99/checkpoint.pth.tar'
)
test_cfg = dict(
    min_score=0.90, # 0.88
    min_area=260,
    min_kernel_area=2.6, # 2.6
    scale=4,
    bbox_type='poly',
    result_path='outputs/submit_ctw/'
)
