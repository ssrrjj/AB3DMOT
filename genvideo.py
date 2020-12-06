import os
import sys
import cv2
def genvideo(image_dir, save_name):
    num_imgs = len(os.listdir(image_dir))
    imgs = []
    for img_idx in range(num_imgs):
        img_path = os.path.join(image_dir, '%06d.jpg'%img_idx)
        imgs.append(cv2.imread(img_path))
    # video = cv2.VideoWriter("%s.mp4"%save_name, cv2.VideoWriter_fourcc(*'XVID'), 10, (1242,374))
    video = cv2.VideoWriter("%s.avi"%save_name, cv2.VideoWriter_fourcc(*'XVID'), 10, (1242,374))
    for image in imgs:
       # plt.imshow(image)
        video.write(image)
    video.release()
    
if __name__=='__main__':
    image_dir = sys.argv[1]
    save_name = sys.argv[2]
    genvideo(image_dir, save_name)
    