import os
import json
import csv
import collections

import subprocess


def TBrain_train():

    root = '/root/Storage/datasets/TBrain_2/train/'

    for count, file_name in enumerate(os.listdir(root + 'json/')):

        with open(root + 'json/' + file_name) as jf:

            if count < 3565:
                gt_name = root + 'train_gts/' + file_name.replace('.json', '.jpg.txt')
            else:
                gt_name = root + 'test_gts/' + file_name.replace('.json', '.jpg.txt')
                subprocess.run(["mv", root + 'train_images/' + file_name.replace('.json', '.jpg'), root + 'test_images/'])

            with open(gt_name, 'w') as gf:

                text = json.load(jf)

                for t in text['shapes']:

                    if t['group_id'] == 0 or t['group_id'] == 2:
                        word = t['label'].replace('#', '')
                        coord = t['points']
                        gf.write(','.join([str(x) + ',' + str(y) for x, y in coord]) + ',' + word + '\n')

def TBrain_test():

    root = '/root/Storage/datasets/TBrain_2/train/'

    for file in os.listdir(root + 'test_gts'):
        src = root + 'test_gts/' + file
        tar = root + 'test_gts/gt_' + file.lstrip('gt') + '.txt'
        # print(tar)
        subprocess.run(["mv", src, tar])

if __name__ == '__main__':
    # TBrain_train()
    TBrain_test()