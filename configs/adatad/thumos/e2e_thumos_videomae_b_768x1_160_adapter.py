_base_ = ["e2e_thumos_videomae_s_768x1_160_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(embed_dims=768, depth=12, num_heads=12),
        custom=dict(pretrain="/root/autodl-tmp/adatad/pretrain/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"),
    ),
)

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)]))

work_dir = "exps/thumos/adatad0215.2/e2e_actionformer_videomae_b_768x1_160_adapter"
