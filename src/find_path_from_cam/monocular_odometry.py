import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


class MonoVideoOdometery(object):
    def __init__(self, 
                img_file_path,
                pose_file_path,
                focal_length = 718.8560,
                pp = (607.1928, 185.2157), 
                lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)), 
                detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
        '''
        Arguments:
            img_file_path {str} -- File path that leads to image sequences
            pose_file_path {str} -- File path that leads to true poses from image sequence
        
        Keyword Arguments:
            focal_length {float} -- Focal length of camera used in image sequence (default: {718.8560})
            pp {tuple} -- Principal point of camera in image sequence (default: {(607.1928, 185.2157)})
            lk_params {dict} -- Parameters for Lucas Kanade optical flow (default: {dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))})
            detector {cv2.FeatureDetector} -- Most types of OpenCV feature detectors (default: {cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)})
        
        Raises:
            ValueError -- Raised when file either file paths are not correct, or img_file_path is not configured correctly
        '''

        self.file_path = img_file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3))
        self.r = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 1))
        self.p = np.zeros(shape=(3, 1))
        self.id = 0
        self.n_features = 0
        self.prevDir = np.zeros(shape=(3, 1))

        try:
            if not all([".jpg" in x for x in os.listdir(img_file_path)]):
                raise ValueError("img_file_path is not correct and does not exclusively .jpg files")
        except Exception as e:
            print(e)
            raise ValueError("The designated img_file_path does not exist, please check the path and try again")

        try:
            with open(pose_file_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError("The pose_file_path is not valid or did not lead to a txt file")

        self.process_frame()


    def hasNextFrame(self):
        return self.id


    def detect(self, img):
        '''Used to detect features and parse into useable format

        
        Arguments:
            img {np.ndarray} -- Image for which to detect keypoints on
        
        Returns:
            np.array -- A sequence of points in (x, y) coordinate format
            denoting location of detected keypoint
        '''

        p0 = self.detector.detect(img)
        
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)


    def visual_odometery(self):
        '''
        Used to perform visual odometery. If features fall out of frame
        such that there are less than 2000 features remaining, a new feature
        detection is triggered. 
        '''
        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)

        if self.p0.shape[0] > 10 or self.id < 2:
            # Calculate optical flow between frames, st holds status
            # of points from frame to frame
            self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
            
            # cameraMatrix = np.array([[572.0693897747975, 0.0, 302.4507642831329], 
            #                 [0.0, 572.0092019521778, 249.7364389652488], 
            #                 [0.0, 0.0, 1.0],])
            
            # dist = np.array([0.13630182840533106, -0.5755884955905456, 0.0027058610814106043, -0.004054602740872948, 0.6278866433969468])
            
            # Save the good points from the optical flow
            self.good_old = self.p0[st == 1]
            self.good_new = self.p1[st == 1]

            # self.good_old = cv2.undistortPoints(self.good_old, cameraMatrix=cameraMatrix, distCoeffs=dist)
            # self.good_new = cv2.undistortPoints(self.good_new, cameraMatrix=cameraMatrix, distCoeffs=dist)
            
            E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.99999, 1, None)
            _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, focal=self.focal, pp=self.pp, mask=None)
            # If the frame is one of first two, we need to initalize
            # our t and R vectors so behavior is different
            if self.id < 2:
                self.r = R
                self.t = t
            else:
                
                absolute_scale = self.get_absolute_scale()
                if (abs(t[1][0]) > abs(t[0][0]) and abs(t[1][0]) > abs(t[2][0])):
                    self.prevDir = self.r.dot(t)
                    self.p = self.p + absolute_scale*self.prevDir
                    self.r = self.r.dot(R)
                else:
                    self.p = self.p + absolute_scale*self.prevDir
                    
            # Save the total number of good features
            self.n_features = self.good_new.shape[0]
        else:
            absolute_scale = self.get_absolute_scale()
            self.p = self.p + absolute_scale*self.prevDir
                

        


    def get_mono_coordinates(self):
        # We multiply by the diagonal matrix to fix our vector
        # onto same coordinate axis as true values
        diag = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
        adj_coord = np.matmul(diag, self.p)

        return adj_coord.flatten()


    def get_true_coordinates(self):
        '''Returns true coordinates of vehicle
        
        Returns:
            np.array -- Array in format [x, y, z]
        '''
        return self.true_coord.flatten()


    def get_absolute_scale(self):
        '''Used to provide scale estimation for mutliplying
           translation vectors
        
        Returns:
            float -- Scalar value allowing for scale estimation
        '''
        pose = self.pose[int(self.id - 1)].strip().split(",")
        x_prev = float(pose[0])
        y_prev = float(pose[1])
        z_prev = float(pose[2])
        pose = self.pose[int(self.id)].strip().split(",")
        x = float(pose[0])
        y = float(pose[1])
        z = float(pose[2])

        true_vect = np.array([[x], [y], [z]])
        self.true_coord = true_vect
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
        
        return np.linalg.norm(true_vect - prev_vect)


    def process_frame(self):
        '''Processes images in sequence frame by frame
        '''
        if self.id < 2:
            # self.old_frame = cv2.imread(self.file_path+"/ezgif-frame-" +str(1).zfill(3)+'.jpg', 0)
            # self.current_frame = cv2.imread(self.file_path +"/ezgif-frame-" + str(2).zfill(3)+'.jpg', 0)
            # self.visual_odometery()
            # self.id = 3
            self.old_frame = cv2.imread(self.file_path+"/" +str(0).zfill(6)+'.jpg', 0)
            self.current_frame = cv2.imread(self.file_path +"/" + str(1).zfill(6)+'.jpg', 0)
            self.visual_odometery()
            self.id = 2
        else:
            # self.old_frame = self.current_frame
            # self.current_frame = cv2.imread(self.file_path +"/ezgif-frame-" + str(self.id).zfill(3)+'.jpg', 0)
            # self.visual_odometery()
            # self.id += 1
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(self.file_path +"/" + str(self.id).zfill(6)+'.jpg', 0)
            self.visual_odometery()
            self.id += 1




if __name__ == '__main__':
    img_path = './data/images'
    pose_path ='./data/pathlogs/logs_run0.csv'
    max = 539
    focal = (572.0693897747975+572.0092019521778)/2
    pp = (302.4507642831329, 249.7364389652488)
    
    # img_path = './data/phoneImages20'
    # pose_path ='./data/pathlogs/logs_run0.csv'
    # max = 200
    # focal = 1
    # pp = (0, 0)
    
    R_total = np.zeros((3, 3))
    t_total = np.empty(shape=(3, 1))
    
    flag = True

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (21,21),
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


    # Create some random colors
    color = np.random.randint(0,255,(5000,3))

    vo = MonoVideoOdometery(img_path, pose_path, focal, pp, lk_params)
    traj = np.zeros(shape=(600, 800, 3))

    while(max > vo.hasNextFrame()):

        frame = vo.current_frame
        cv2.imshow('frame', frame)
        k = cv2.waitKey(100)
        if k == 27:
            break

        vo.process_frame()

        mono_coord = vo.get_mono_coordinates()
        true_coord = vo.get_true_coordinates()

        print("MSE Error: ", np.linalg.norm(mono_coord - true_coord))
        print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in mono_coord]))
        print("true_x: {}, true_y: {}, true_z: {}".format(*[str(pt) for pt in true_coord]))

        draw_x, draw_y, draw_z = [int(round(200*x)) for x in mono_coord]
        true_x, true_y, true_z = [int(round(200*x)) for x in true_coord]

        traj = cv2.circle(traj, (true_x+300, true_y+200), 1, list((0, 0, 255)), 4)
        
        traj = cv2.circle(traj, (draw_x+300, draw_y+200), 1, list((0, 255, 0)), 4)

        cv2.putText(traj, 'Actual Position:', (140, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
        cv2.putText(traj, 'Red', (270, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
        cv2.putText(traj, 'Estimated Odometry Position:', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
        cv2.putText(traj, 'Green', (270, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)

        cv2.imshow('trajectory', traj)
    cv2.imwrite("./trajectory.png", traj)

    cv2.destroyAllWindows()