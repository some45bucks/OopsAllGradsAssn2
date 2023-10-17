import rclpy
from rclpy.node import Node
import os
import csv
from std_msgs.msg import Float32MultiArray
import smbus
import time
from sensor_msgs.msg import Image
import math
import cv2
from cv_bridge import CvBridge


class Log(Node):
    def __init__(self):
        super().__init__('log')
        self.log_subscription = self.create_subscription(Float32MultiArray, '/log', self.data_callback, 10)
        self.image_subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.x = 0
        self.y = 0
        self.theta = 0
        self.prevTime = 0
        self.imageCount = 0
        self.firstTime = 0
        self.stop = False
        self.bridge = CvBridge()
        self.mem = []
        
        self.startCapture = False

        #find new file for runs
        i = 0
        while os.path.exists(f"data/pathlogs/logs_run{i}.csv"):
            i+=1

        #open file
        self.file = open(f"data/pathlogs/logs_run{i}.csv", 'w')

        self.writer = csv.writer(self.file)

    def data_callback(self, msg):

        v = msg.data[0]
        av = msg.data[1]
        t = msg.data[2] - self.prevTime
        
        if v != 0 or av != 0:
            self.startCapture = True

        if self.startCapture:
            if self.firstTime == 0:
                self.firstTime = msg.data[2]
            #writes data x pos, y pos, angle, and then the time stamp from 0
            self.writer.writerow([self.x,self.y,self.z,self.theta,msg.data[2]-self.firstTime])
            
            xV = v * math.sin(self.theta)
            yV = v * math.cos(self.theta)

            self.x += xV * t
            self.y += yV * t
            self.z = 0
            self.theta += av * t

            #writes data x pos, y pos, angle, and then the time stamp from 0
            self.writer.writerow([self.x,self.y,self.z,self.theta,msg.data[2]])

            self.prevTime = msg.data[2]
        
    def image_callback(self, msg):
        if self.startCapture:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite('data/images/'+str(self.imageCount).zfill(6)+'.jpg', cv_image)
            self.imageCount+=1

            

def main(args=None):
    rclpy.init(args=args)

    subscriber = Log()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt as e:
        print(e)
        subscriber.file.close()
    except Exception as e:
        print(e)
        subscriber.file.close()

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
