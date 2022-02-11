import numpy as np
import random
import cv2
import os
from natsort import natsorted
from tqdm import tqdm

# def draw(img, boxes):
#     mask = np.zeros(img.shape, dtype=np.uint8)
    
#     for box in boxes:
#         rand_r = random.randint(100, 255)
#         rand_g = random.randint(100, 255)
#         rand_b = random.randint(100, 255)
#         mask = cv2.fillPoly(mask, [box], color=(rand_r, rand_g, rand_b))
    
#     img[mask!=0] = (0.6 * mask + 0.4 * img).astype(np.uint8)[mask!=0]
    
#     for box in boxes:
#         cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
    
#     return img

def draw_img():
    output_root = 'outputs/submit_tt_rec/'
    img_root = 'data/total_text/Images/Test/'
    result_root = 'draw_img_result/'

    for file in tqdm(natsorted(os.listdir(output_root))):
        img = cv2.imread(img_root + file.replace('.txt', '.jpg'))
        with open(output_root + file) as result:
            for line in result:
                line_split = line.split(',')

                if line_split[-1].strip() != '###':
                    box = np.array([[int(line_split[i]),int(line_split[i+1])] for i in range(0,len(line_split[:-1]),2)])
                    rgb = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

                    cv2.polylines(img, [box], -1, rgb, 2)
                    cv2.putText(img, line_split[-1].strip(), (box[0,0],box[0,1]-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            cv2.imwrite(result_root + file.replace('.txt', '.jpg'), img)




if __name__ == '__main__':
    draw_img()