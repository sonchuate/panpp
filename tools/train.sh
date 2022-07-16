cd ..

# python train.py config/pan_pp/pan_pp_synth.py
# python train.py config/pan_pp/pan_pp_synth.py --resume /root/Storage/panpp/checkpoints/pan_pp_synth/checkpoint.pth.tar

python train.py config/pan_pp/pan_pp_tt.py
# python train.py config/pan_pp/pan_pp_tt.py --resume /root/Storage/panpp/checkpoints/pan_pp_tt/checkpoint.pth.tar

# python train.py config/pan_pp/pan_pp_tbrain.py
# python train.py config/pan_pp/pan_pp_tbrain.py --resume /root/Storage/panpp/checkpoints/pan_pp_tbrain/checkpoint.pth.tar

# python train.py config/pan_pp/pan_pp_ctw.py
# python train.py config/pan_pp/pan_pp_ctw.py --resume /root/Storage/panpp/checkpoints/pan_pp_ctw/checkpoint.pth.tar

# python train.py config/pan_pp/pan_pp_ic15.py
# python train.py config/pan_pp/pan_pp_ic15.py --resume /root/Storage/panpp/checkpoints/pan_pp_ic15/checkpoint.pth.tar


# python train.py config/pan_pp/pan_pp_tt_joint_train.py \
# --resume checkpoints/pan_pp_tt_joint_train/checkpoint.pth.tar
