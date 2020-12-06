import csv
import numpy as np
import os
import sys

def convert(from_file, to_file, object_id):
    text = []
    with open(from_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            line = np.array([float(x) for x in row])
            if (int(line[1]) == object_id):
                new_line = line.copy()
                if object_id == 1:
                    new_line[1] = 2
                elif object_id == 2:
                    new_line[1] = 1
                new_line[7:10] = line[[9,7,8]]
                new_line[10], new_line[11], new_line[12] = -line[11], -line[12]+line[9]/2, line[10]
                new_line[13] = -line[13]
                new_line = [str(x) for x in new_line]
                new_line = ', '.join(new_line)+'\n'
                text.append(new_line)
    fp = open(to_file,'w')
    for line in text:
        fp.writelines(line)
    fp.close()

if __name__=='__main__':
    get_object_id = {'Pedestrian': 2, 'Car': 1, 'Cyclist': 3}
    from_dir = sys.argv[1]
    
    to_dir = os.path.join("./data/KITTI", sys.argv[2])
    for obj in get_object_id.keys():
        for i in range(21):
            from_file = os.path.join(from_dir, "%04d/%04d.csv" % (i, i))
            to_file = os.path.join(to_dir, "%s/%04d.txt" % (obj, i))
            if not os.path.exists(os.path.join(to_dir, obj)):
                os.makedirs(os.path.join(to_dir,obj))
            if not os.path.exists(from_file):
                print("can't find file",from_file)
                continue
            convert(from_file, to_file, get_object_id[obj])
