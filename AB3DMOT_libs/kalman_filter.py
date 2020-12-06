# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
from filterpy.kalman import KalmanFilter
from oxt_utils import cam_to_w, w_to_cam
class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox3D_and_feature, info):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        bbox3D = bbox3D_and_feature[:7]
        feature = bbox3D_and_feature[7:]
        self.kf = KalmanFilter(dim_x=10, dim_z=7)       
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])

        # # with angular velocity
        # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
        # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
        #                       [0,1,0,0,0,0,0,0,1,0,0],
        #                       [0,0,1,0,0,0,0,0,0,1,0],
        #                       [0,0,0,1,0,0,0,0,0,0,1],  
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0],
        #                       [0,0,0,0,0,0,0,1,0,0,0],
        #                       [0,0,0,0,0,0,0,0,1,0,0],
        #                       [0,0,0,0,0,0,0,0,0,1,0],
        #                       [0,0,0,0,0,0,0,0,0,0,1]])     

        # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
        #                       [0,1,0,0,0,0,0,0,0,0,0],
        #                       [0,0,1,0,0,0,0,0,0,0,0],
        #                       [0,0,0,1,0,0,0,0,0,0,0],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0]])


        # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
        self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

        # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        self.kf.Q[7:, 7:] *= 0.01


        self.kf.x[:7] = bbox3D.reshape((7, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1           # number of total hits including the first detection
        self.hit_streak = 1     # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info        # other info associated

        self.max_history = 15
        self.appear_history = [feature]
        self.score_history = [1]

    def compute_FAS(self, det_feature):
        score_sum = 0
        fas = 0
        if len(self.score_history) == 0:
            return 1
            
        for i, appear in enumerate(self.appear_history):
            scorei = self.score_history[i]
        
            score_sum += scorei
            fas += scorei * np.sum(appear*det_feature)/np.linalg.norm(appear)/np.linalg.norm(det_feature)
        fas = fas/score_sum
        return fas


    def update(self, bbox3D_and_feature, info): 
        """ 
        Updates the state vector with observed bbox.
        """
#         print("update tracker",self.id,"at", self.kf.x.reshape(-1), "with", bbox3D.reshape(-1))

        bbox3D = bbox3D_and_feature[:7]
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1          # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1      # number of continuing hit in the fist time

        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi       
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2

        #########################     # flip

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.info = info

        #compute score
        score = self.compute_FAS(bbox3D_and_feature[7:])
        self.score_history.append(score)
        self.appear_history.append(bbox3D_and_feature[7:])
        if len(self.score_history) > self.max_history:
            self.score_history = self.score_history[1:]
            self.appear_history = self.appear_history[1:]

    def predict(self):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()      
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1].copy()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7, ))