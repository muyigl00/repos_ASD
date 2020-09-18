#!/usr/bin/env python

from __future__ import print_function
import rospy
import tf
from car_demo.msg import Control, Trajectory, ControlDebug,Foo
import numpy as np
from sensor_msgs.msg import JointState, Image
import message_filters
import std_msgs.msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
from simulation_image_helper import SimulationImageHelper

from matplotlib import pyplot as plt # matplotlib
import random
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension
from rospy.numpy_msg import numpy_msg

bridge = CvBridge() 

class ImageHandler:

    def __init__(self):
        self.bridge = CvBridge() 
        # subscribers
        self.imageSub = rospy.Subscriber("/prius/front_camera/image_raw", Image, self.ImgHandlerCallback, queue_size=1)
        # The queue_size in the last line is really important! The code here is not thread-safe, so
        # - by all cost - you have to avoid that the callback function is called multiple times in parallel.

        # publishers
        self.pubDbgImage = rospy.Publisher("lane_detection_dbg_image", Image, queue_size=1)
	self.mypub = rospy.Publisher('plot_topic', Foo)

        self.H_I = np.matrix([[-7.74914499e-03, 3.95733793e-18, 3.10353257e+00],
                              [8.56519716e-18,  9.42313768e-05, -1.86052093e+00],
                              [2.57498016e-18, -2.73825295e-03,  1.00000000e+00]])
        self.H_I = self.H_I.reshape(3, 3)
        self.H = self.H_I.I

        X_infinity = np.matrix([0, 10000, 1]).reshape(3, 1)
        self.yHorizon = self.H*X_infinity
        self.yHorizon = int((self.yHorizon[1] / self.yHorizon[2]).item())


    def image2road(self, pts2d):
        """
        Transform a point on the road surface from the image to 3D coordinates (in DIN70000, i.e. X to front, Y to left, 
        origin in center of vehicle at height 0).
        """
        N, cols = pts2d.shape
        if (cols == 1) and (N == 2):
            # if just two coordinates are given, we make sure we have a (2x1) vector
            pts2d = pts2d.T
        assert cols == 2, "pts2d should be a Nx2 numpy array, but we obtained (%d, %d)" % (
            N, cols)

        X = self.H_I * np.hstack((pts2d, np.ones((pts2d.shape[0], 1)))).T
        X = X/X[2, :]
        X = X.T

        # the camera is "self.C[2]" in front of the vehicle's center
        return np.hstack((X[:, 1], -X[:, 0]))


    def road2image(self, ptsRd, imSize=None):
        """
        Transform a 3d point on the road surface (in DIN70000, i.e. X to front, Y to left) to image coordinates.
        """
        N, cols = ptsRd.shape
        if (cols == 1) and (N == 2):
            # if just two coordinates are given, we make sure we have a (2x1) vector
            ptsRd = ptsRd.T
            N = cols
        assert cols == 2, "ptsRd should be a Nx2 numpy array, but we obtained (%d, %d)" % (
            N, cols)

        # go back from DIN70000 to our image coordinate system
        pts = np.hstack((-ptsRd[:, 1].reshape((-1, 1)),
                         ptsRd[:, 0].reshape((-1, 1)), np.ones((N, 1))))
        x = self.H * pts.T
        x = x/x[2, :]
        x = x.T

        if imSize != None:
            valid = np.logical_and((x[:, 0] >= 0), (x[:, 1] >= 0))
            valid = np.logical_and(valid, (x[:, 0] < imSize[1]))
            valid = np.logical_and(valid, (x[:, 1] < imSize[0]))
            valid = np.nonzero(valid)[0].tolist()
            x = x[valid, :]

        return np.hstack((x[:, 0], x[:, 1]))


    def ImgHandlerCallback(self,message):
	try:
		cv_image=bridge.imgmsg_to_cv2(message, "mono8")
	except CvBridgeError as e:
		rospy.logerr("Error in imageCallback: %s, Raghid", e)

	# Das bild kleiner schneiden um den relevanten teil hervor zu heben
	cv_image = cv_image[385:800,:]

	# die spurlinien ausduennen um genauere werte zu erhalten
	kernel = np.ones((3,3),np.uint8)
	erosion =cv2.erode(cv_image,kernel,iterations =1)

	#dafur sorgen dass das fahrzeug ausgeblendet wird
	cv_image_white = ( erosion > 170) 

	#bild jovertieren von bool zu int
	y = cv_image_white.astype(int)
	cv_image_white = np.uint8(y * 255)

	#gradient bildung in x richtung um seiten punkte zu erhalten
	Ix = cv2.Sobel(cv_image_white ,cv2.CV_64F,0,1,ksize=5 )
	cv_image_white = np.uint8(Ix * 255)

	# Wir nehmen alle Weissen Punkte und fuegen diese in einen Array
	n=0
	x1 = []
	x2 = []
	dummy = np.argwhere(cv_image_white > 170)
	x = dummy
	x[:,0] = dummy[:,1]
	x[:,1] = dummy[:,0]+385


	"""for i in range(cv_image_white.shape[0]):
		for m in range(cv_image_white.shape[1]):
			if(cv_image_white[i][m] > 170 and n % 9==0):
				x1.append(385+i)
				x2.append(m) 			
			n=n+1
        """	

	# Zusammenfuegen beider Array
	#x1=np.array(x1)
	#x2=np.array(x2)
	#x = np.vstack((x2, x1)).T
	
	# hier konventieren wir die 2D-Bildpunkte in 3D um
	M = self.image2road(x)

	# hier publishen wir die x,y Werte an die plot_node
	msg_to_send = Foo()
	msg_to_send.some_y = -M[:,1]
	msg_to_send.some_x = M[:,0]
	self.mypub.publish(msg_to_send)

	#bild von Open_CV2 in Ros_message format zu konvertieren
	try:
		ROS_image=bridge.cv2_to_imgmsg(cv_image_white, "mono8")
	except CvBridgeError as e:
		rospy.logerr("Error in imageCallback: %s, Raghid", e)
	self.pubDbgImage.publish(ROS_image)


if __name__ == '__main__':

    rospy.init_node('ImgPublisher')
    rate = rospy.Rate(10.0)


    imageHandler = ImageHandler()
    rospy.spin()
