cd ..
python test.py \
config/pan_pp/pan_pp_tt.py \
/root/Storage/panpp/checkpoints/pan_pp_tt/checkpoint.pth.tar
cd eval/
. ~/anaconda3/etc/profile.d/conda.sh
conda activate py27
sh eval_tt.sh
