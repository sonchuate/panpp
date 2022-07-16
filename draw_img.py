import numpy as np
import random
import cv2
import os
from natsort import natsorted
from tqdm import tqdm
import csv
from collections import defaultdict

def draw_img_tt_gt():
    output_root = '/root/Storage/datasets/total_text/test/test_gts/'

    img_root = 'data/total_text/Images/Test/'
    result_root = 'draw_img_result/'

    for file in tqdm(natsorted(os.listdir(output_root))):
        img = cv2.imread(img_root + file.replace('.txt', ''))
        with open(output_root + file) as result:
            for line in result:
                line_split = line.split(',')

                box = np.array([[int(line_split[i]),int(line_split[i+1])] for i in range(0,len(line_split[:8]),2)])
                # rgb = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

                cv2.polylines(img, [box], -1, (0,0,255), 3)

            cv2.imwrite(result_root + file.replace('.txt', ''), img)


def draw_img_tt_result():
    output_root = 'outputs/submit_tt/'

    img_root = 'data/total_text/Images/Test/'
    result_root = 'draw_img_result/'

    for file in tqdm(natsorted(os.listdir(output_root))):
        img = cv2.imread(img_root + file.replace('.txt', '.jpg'))
        with open(output_root + file) as result:
            for line in result:
                line_split = line.split(',')

                box = np.array([[int(line_split[i+1]),int(line_split[i])] for i in range(0,len(line_split),2)])
                # rgb = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

                cv2.polylines(img, [box], -1, (0,0,255), 3)

            cv2.imwrite(result_root + file.replace('.txt', '.jpg'), img)


def draw_img_tbrain():
    output_root = 'outputs/submit_tbrain/'
    img_root = '/root/Storage/datasets/TBrain_2/train/test_images/'
    result_root = 'draw_img_result/'

    for file in tqdm(natsorted(os.listdir(output_root))):
        img_file_name = file.replace('.txt','.jpg').lstrip('res_').lstrip('gt_')
        img = cv2.imread(img_root + img_file_name)
        with open(output_root + file) as result:
            for line in result:
                line_split = line.split(',')
                box = np.array([[int(line_split[i]),int(line_split[i+1])] for i in range(0,len(line_split[:-1]),2)])
                # rgb = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

                cv2.polylines(img, [box], -1, (0,0,255), 2)
            cv2.imwrite(result_root + file.replace('.txt', '.jpg'), img)

def draw_public_img():
    img_root = '/root/Storage/datasets/TBrain_2/public/img_public/'
    csv_root = '/root/Storage/datasets/TBrain_2/public/Task2_Public_String_Coordinate.csv'
    result_root = 'draw_img_result/'

    d = defaultdict(list)

    with open(csv_root) as csvfile:

        rows = csv.reader(csvfile)

        for row in rows:
            d[row[0]].append(row[1:])

    for k, v in tqdm(d.items()):
        
        img = cv2.imread(img_root + k + '.jpg')

        for line in v:        
            box = np.array([[int(line[i]),int(line[i+1])] for i in range(0,len(line[:-1]),2)])
            rgb = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            cv2.polylines(img, [box], -1, rgb, 2)
        
        cv2.imwrite(result_root + k + '.jpg', img)



if __name__ == '__main__':
    if not os.path.isdir("draw_img_result"):
        os.mkdir("draw_img_result")
    # draw_img_tt_gt()
    draw_img_tt_result()
    # draw_img_tbrain()
    # draw_public_img()