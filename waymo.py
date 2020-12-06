import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import matplotlib.pyplot as plt
class WaymoObject(object):
    def __init__(self, x,y,z,l,w,h,heading,id,type):
        self.t = np.array([x,y,z])
        self.l = l
        self.w = w
        self.h = h
        self.heading = heading
        self.id = id
        self.type = type

def get_box_transformation_matrix(box):
    """Create a transformation matrix for a given label box pose."""

    tx,ty,tz = box.t
    c = math.cos(box.heading)
    s = math.sin(box.heading)

    sl, sh, sw = box.l, box.h, box.w

    return np.array([
        [ sl*c,-sw*s,  0,tx],
        [ sl*s, sw*c,  0,ty],
        [    0,    0, sh,tz],
        [    0,    0,  0, 1]])

def get_3d_box_projected_corners(vehicle_to_image, label):
    """Get the 2D coordinates of the 8 corners of a label's 3D bounding box.
    vehicle_to_image: Transformation matrix from the vehicle frame to the image frame.
    label: The object label
    """

    # Get the vehicle pose
    box_to_vehicle = get_box_transformation_matrix(label)

    # Calculate the projection from the box space to the image space.
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)


    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)

    return vertices

def compute_2d_bounding_box(img_or_shape,points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    if isinstance(img_or_shape,tuple):
        shape = img_or_shape
    else:
        shape = img_or_shape.shape

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    x1 = min(max(0,x1),shape[1])
    x2 = min(max(0,x2),shape[1])
    y1 = min(max(0,y1),shape[0])
    y2 = min(max(0,y2),shape[0])

    return (x1,y1,x2,y2)

def draw_3d_box(img, vehicle_to_image, label, colour=(255,128,128)):
    """Draw a 3D bounding from a given 3D label on a given "img". "vehicle_to_image" must be a projection matrix from the vehicle reference frame to the image space.
    draw_2d_bounding_box: If set a 2D bounding box encompassing the 3D box will be drawn
    """
    import cv2

    vertices = get_3d_box_projected_corners(vehicle_to_image, label)

    if vertices is None:
        # The box is not visible in this image
        return


    # Draw the edges of the 3D bounding box
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1,idx2 in [((0,k,l),(1,k,l)), ((k,0,l),(k,1,l)), ((k,l,0),(k,l,1))]:
                cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)
    # Draw a cross on the front face to identify front & back.
    for idx1,idx2 in [((1,0,0),(1,1,1)), ((1,1,0),(1,0,1))]:
        cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)




def decode_image(camera):
    """ Decode the JPEG image. """

    from PIL import Image
    return np.array(Image.open(io.BytesIO(camera.image)))

def get_image_transform(camera_calibration):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """

    # TODO: Handle the camera distortions
    extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4,4)
    intrinsic = camera_calibration.intrinsic

    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0  0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))
    return vehicle_to_image
def display_labels_on_image(camera_calibration, camera, labels):
    # Get the image transformation matrix
    vehicle_to_image = get_image_transform(camera_calibration)

    # Decode the JPEG image
    img = np.array(tf.image.decode_jpeg(camera.image))
    
    # Draw all the groundtruth labels
    for label in labels:
        draw_3d_box(img, vehicle_to_image, label)

    # Display the image
    return img
    #cv2.waitKey(display_time)
def read_objs(file):
    objs = []
    with open(file, 'r') as fp:
        for line in fp.readlines():
            data = line.split()
            data[1:] = [float(x) for x in data[1:]]
            h = data[8] # box height
            w = data[9] # box width
            l = data[10] # box length (in meters)
            t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
            heading = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
            if len(data) > 16: 
                id = int(data[16])
            obj = WaymoObject(t[0],t[1],t[2],l,w,h,heading,id,data[0])
            objs.append(obj)
    return objs



filename = "/waymo-od/validation/segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord"
dataset = tf.data.TFRecordDataset(filename, compression_type='')
i = 0
trk_dir = "/team1/codes/individual/vkonduru/AB3DMOT/results/waymo_25_5_val/VEHICLE/trk_withid/segment-10203656353524179475_7625_000_7645_000_with_camera_labels"
imgs = []
for data in dataset:
    frame = open_dataset.Frame()
    trk_file = os.path.join(trk_dir, "%06d.txt"%i)
    i+=1
    
    objs = read_objs(trk_file)
    print(len(frame.images))
    for index, camera in enumerate(frame.images):
        camera_calibration = frame.context.camera_calibrations[index]
        img = display_labels_on_image(camera_calibration, camera, objs)
        imgs.append(img)
    break
