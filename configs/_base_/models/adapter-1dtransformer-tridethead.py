model = dict(
    type="ActionFormer",
    projection=
        dict(
            type="Conv1DTransformerProj",
            in_channels=768,
            out_channels=512,
            arch=(2, 2, 5),  # layers in embed / stem / branch
            conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
            norm_cfg=dict(type="LN"),
            attn_cfg=dict(n_head=4, n_mha_win_size=-1),
            path_pdrop=0.1,
            use_abs_pe=False,
            max_seq_len=768,
            transformer1d_rel_pos_cfg=dict(
            in_channels=768,  
            out_channels=768,  
            encoder_layer_cfg=dict(
                dim=768,  
                num_heads=6,  
                drop_path=0.1,  
            ),
            num_layers=3,  
            )
        ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="TriDetHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        boundary_kernel_size=3,
        iou_weight_power=0.2,
        num_bins=16,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
            iou_rate=dict(type="GIOULoss"),
        ),
    ),
)
