cd ..
python test.py \
config/pan_pp/pan_pp_ctw.py \
/root/Storage/panpp/checkpoints/pan_pp_ctw/checkpoint.pth.tar
cd eval/
. ~/anaconda3/etc/profile.d/conda.sh
conda activate py27
sh eval_ctw.sh
