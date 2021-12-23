cd ..
python test.py config/pan_pp/pan_pp_r18_ic15_joint_train.py trained/panpp_r18_joint_train.pth.pth.tar
cd eval/
. ~/anaconda3/etc/profile.d/conda.sh
conda activate py27
sh eval_ic15.sh
