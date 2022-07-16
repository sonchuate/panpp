import os
import subprocess
from natsort import natsorted

def eval_tt():

    config = 'pan_pp_ctw.py'
    
    # checkpoints = 'pan_pp_tt'
    checkpoints = 'pan_pp_ctw'
    # checkpoints = 'pan_pp_synth'
    # checkpoints = 'pan_pp_ic15'


    model_root = '/root/Storage/panpp/checkpoints/'+ checkpoints +'/'
    print(model_root)
    model_list = natsorted(os.listdir(model_root), reverse=True)
    
    if 'cjg.json' in model_list:
        model_list.remove('cjg.json')
    if 'checkpoint.pth.tar' in model_list and len(model_list) != 1:
        model_list.remove('checkpoint.pth.tar')


    try:
        for model in model_list:
            print(model)

            os.chdir('/root/Storage/panpp')

            subprocess.call([
                '/root/anaconda3/envs/pan/bin/python', 'test.py', 'config/pan_pp/' + config,
                model_root + model
            ])

            os.chdir('/root/Storage/panpp/eval/ctw')

            subprocess.call([
                '/root/anaconda3/envs/py27/bin/python', 'eval.py'
            ])

            # break

    except:
        print('error')


if __name__ == '__main__':
    eval_tt()
