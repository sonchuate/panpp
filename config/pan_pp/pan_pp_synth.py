model = dict(
    # type='PAN',
    type='PAN_PP_V2',
    backbone=dict(
        # type='resnet18',
        type='resnet18_csp',
        # type='resnet18_fusion',
        pretrained=True
    ),
    neck=dict(
        # type='FPEM_v1',
        type='FPN_v3_3',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        # type='PA_Head',
        # type='PAN_PP_DetHead_v2',
        type='PAN_PP_DetHead_v2_1',
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
        use_coordconv=False,
    )
)
data = dict(
    batch_size=8,
    num_workers=8,
    train=dict(
        type='PAN_PP_Synth',
        is_transform=True,
        img_size=640,
        short_size=640,
        kernel_scale=0.5,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=4,
    optimizer='Adam'
)
