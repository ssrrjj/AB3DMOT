"""
    Mean feature vector for each class is : 64 dim vector with 100 at the object id index and 0 everywhere
    For all new occurrences, we add gaussian noise on top of mean
    So for now, use data with less than 64 objects
"""
import argparse
from pathlib import Path
import numpy as np
import os
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--gt_file', type=str, default='./evaluation/label/0000.txt',
                        help='specify the gt file to simulate over')
    parser.add_argument('--save_dir', type=str, default='./data/KITTI/simulated/',
                        help='specify the directory to save')

    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    text = []

    det_id2str = {
        'Pedestrian':1, 
        'Car':2, 
        'Van':2, 
        'Cyclist':3
        }

    with open(args.gt_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        counter = 0
        for row_list in reader:
            counter += 1
            print(counter)
            # 0 0 Van 0 0 -1.793451 296.744956 161.752147 455.226042 292.372804 2.000000 1.823255 4.433886 -4.552284 1.858523 13.410495 -2.115488
            """
            Frame	Type	2D BBOX (x1, y1, x2, y2)	Score	3D BBOX (h, w, l, x, y, z, rot_y)	Alpha
            0	2 (car)	726.4, 173.69, 917.5, 315.1	13.85	1.56, 1.58, 3.48, 2.57, 1.57, 9.72, -1.56	-1.82
            """
            # Convert to the detections format
            new_line = []

            new_line.append(row_list[0]) # Frame Id
            if row_list[2] not in det_id2str.keys():
                continue

            object_id = det_id2str[row_list[2]]
            new_line.append(object_id) # Object ID
            
            for i in range(5): # 2d properties
                new_line = new_line + [0]

            # 3d properties
            for i in range(7):
                new_line = new_line + [row_list[10 + i]]

            new_line = new_line + [0] # Score

            # Adding similarity descriptor
            mean = np.zeros((64))
            mean[object_id] = 100
            cov = np.eye(64)
            feature_vec = np.random.multivariate_normal(mean, cov)

            new_line = new_line + list(feature_vec)

            new_line = [str(x) for x in new_line]
            new_line = ', '.join(new_line)+'\n'
            text.append(new_line)

    file_name = args.gt_file.split("/")[-1]
    fp = open(args.save_dir + "0000_all.txt",'w')
    for line in text:
        fp.writelines(line)
    fp.close()

if __name__ == '__main__':  
    main()