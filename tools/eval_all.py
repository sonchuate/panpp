import os
import subprocess
from natsort import natsorted

def eval_tt():

    model_root = '/root/Storage/panpp/checkpoints/pan_pp_tt/'
    print(model_root)
    model_list = natsorted(os.listdir(model_root), reverse=True)
    model_list.remove('checkpoint.pth.tar')

    try:
        for model in model_list:
            print(model)

            os.chdir('/root/Storage/panpp')

            subprocess.call([
                '/root/anaconda3/envs/pan/bin/python', 'test.py', 'config/pan_pp/pan_pp_tt.py',
                model_root + model
            ])

            os.chdir('/root/Storage/panpp/eval/tt')

            subprocess.call([
                '/root/anaconda3/envs/py27/bin/python', 'Deteval.py'
            ])

            # break

    except:
        print('error')


if __name__ == '__main__':
    eval_tt()
