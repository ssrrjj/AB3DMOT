import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image
from AB3DMOT_libs.kitti_utils import read_label, compute_box_3d, draw_projected_box3d, Calibration, Object3d
def visgroundtruth(data_root, seq, object_name):
    width = 1242
    height = 374
    object_to_id = {"Car": 2, "Cyclist": 3, "Pedestrian": 1, "Van": 2}
    objects = {'Car',  "Cyclist", "Pedestrian"}
    def show_image_with_boxes(img, objects_res, object_gt, calib, save_path, height_threshold=0):
        img2 = np.copy(img) 
        # for each object, compute the bouding box in 2D image. Use the same color for objects of same obj.id. 
        for obj in objects_res:
            box3d_pts_2d, _ = compute_box_3d(obj, calib.P) # calib.P is the projection matrix from camera coord to image coord
            color_tmp = tuple([255,0,0])
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=color_tmp)
            text = 'ID: %d' % obj.id
            if box3d_pts_2d is not None:
                img2 = cv2.putText(img2, text, (int(box3d_pts_2d[4, 0]), int(box3d_pts_2d[4, 1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color_tmp) 

        img = Image.fromarray(img2)
        img = img.resize((width, height))
        img.save(save_path)
    image_dir = os.path.join(data_root, seq)
    print(image_dir)
    calib_file = os.path.join('data/KITTI/resources/training/', 'calib/%s.txt' % seq) 

    save_3d_bbox_dir = os.path.join('./results','det_trk_gt_vis/%s' % seq)
    print(save_3d_bbox_dir)
    if not os.path.exists(save_3d_bbox_dir):
        os.makedirs(save_3d_bbox_dir)

    seq_file = os.path.join('evaluation/label',"%s.txt"%seq)
    with open(seq_file,"r") as fp:
        
        lines = fp.readlines()
    seq_dets = []
    for line in lines:
        obj = line.split()
        seq_dets.append(obj)
    seq_dets = np.array(seq_dets)
    if len(seq_dets.shape) == 1: seq_dets = np.expand_dims(seq_dets, axis=0) 	
    if seq_dets.shape[1] == 0:
        return
    calib_tmp = Calibration(calib_file)
    # loop over frame
    min_frame, max_frame = int(seq_dets[:, 0].astype(int).min()), int(seq_dets[:, 0].astype(int).max())
    print(min_frame, max_frame)
    for frame in range(min_frame, max_frame + 1):
        try:
            img_tmp = np.array(Image.open(os.path.join(image_dir,"%06d.png"%frame)))
        except:
            break
        # logging
        print_str = 'processing %s:  %d/%d   \r' % (seq,  frame, max_frame)
        print(print_str)

        dets = seq_dets[seq_dets[:,0] == str(frame)]           
# frame 0, track id 1, type 2, truncated 3, occluded 4, alpha 5, bbox 6 7 8 9, dimemsions 10 11 12, location 13 14 15,
# rotation_y 16
        
        objects = []
        for i, det in enumerate(dets):
            if det[2] in object_to_id:
                str_to_srite = '%s -1 -1 %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % (det[2], det[5], 
                                det[6], det[7], det[8], det[9], det[10], det[11], det[12], det[13], det[14], det[15], det[16], "1", det[1])
#             str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % ("Car", det[14],
#                     det[2], det[3], det[4], det[5], 
#                     det[7], det[8], det[9], det[10], det[11], det[12], det[13], det[6], 1)
                objects.append(Object3d(str_to_srite))
        save_path_tmp = os.path.join(save_3d_bbox_dir, '%06d.jpg' % frame)
        show_image_with_boxes(img_tmp, objects, [], calib_tmp, save_path_tmp)

seqs = ["%04d"%i for i in range(16,21)]
data_root = "/shared/kitti-odometry/training/image_02/"
calib = 'data/KITTI/resources/training/'
for seq in seqs:
	visgroundtruth(data_root, seq, "Car")	
