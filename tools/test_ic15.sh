cd ..
python test.py \
config/pan_pp/pan_pp_ic15.py \
/root/Storage/panpp/checkpoints/pan_pp_r18_ic15_exp41/checkpoint_590ep.pth.tar
cd eval/
. ~/anaconda3/etc/profile.d/conda.sh
conda activate py27
sh eval_ic15.sh
