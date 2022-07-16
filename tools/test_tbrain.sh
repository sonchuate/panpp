cd ..
python test.py \
config/pan_pp/pan_pp_tbrain.py \
/root/Storage/panpp/checkpoints/pan_pp_tbrain_exp119/checkpoint_570ep.pth.tar
cd eval/tbrain/
. ~/anaconda3/etc/profile.d/conda.sh
conda activate py27
python script.py -g=gt.zip -s=../../outputs/submit_tbrain.zip