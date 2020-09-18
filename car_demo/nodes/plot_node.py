#!/usr/bin/env python 

from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import rospy
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension
from rospy.numpy_msg import numpy_msg
from car_demo.msg import Foo, Regelung
import random
from nav_msgs.msg import Odometry

import rospy
import tf
from car_demo.msg import Control, Trajectory, ControlDebug
import numpy as np
from sensor_msgs.msg import JointState, Image
import message_filters
import std_msgs.msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
from simulation_image_helper import SimulationImageHelper




def LS_lane_fit(pL, pR):
    """
    LS estimate for lane coeffients z=(W, Y_offset, Delta_Phi, c0)^T.
    
    Args:
        pL: [NL, 2]-array of left marking positions (in DIN70000) 
        pR: [NR, 2]-array of right marking positions (in DIN70000)
    
    Returns:
        Z: lane coeffients (W, Y_offset, Delta_Phi, c0)
    """
    
    H = np.zeros((pL.shape[0]+pR.shape[0], 4)) # design matrix
    Y = np.zeros((pL.shape[0]+pR.shape[0], 1)) # noisy observations
    
    # fill H and Y for left line points
    for i in range(pL.shape[0]):
        u, v = pL[i,0], pL[i,1]
        u2 = u*u
        H[i, :] = [0.5, -1, -u, 1.0/2.0 * u2]
        Y[i] = v

    # fill H and Y for right line points
    for i in range(pR.shape[0]):
        u, v = pR[i,0], pR[i,1]
        u2 = u*u
        u3 = u2*u 
        H[pL.shape[0]+i, :] = [-0.5, -1, -u, 1.0/2.0 * u2]
        Y[pL.shape[0]+i] = v

    Z = np.dot(np.linalg.pinv(H), Y)
    
    return Z



def LS_lane_compute(Z, maxDist=60, step=0.5):
    """
    Compute lane points from given parameter vector.
    
    Args;
        Z: lane coeffients (W, Y_offset, Delta_Phi, c0)
        maxDist[=60]: distance up to which lane shall be computed
        step[=0.5]: step size in x-direction (in m)
       
    Returns:
        (x_pred, yl_pred, yr_pred): x- and y-positions of left and 
            right lane points
    """
    x_pred = np.arange(0, maxDist, step)
    yl_pred = np.zeros_like(x_pred)
    yr_pred = np.zeros_like(x_pred)

    for i in range(x_pred.shape[0]):
        u = x_pred[i]
        u2 = u*u
        yl_pred[i] = np.dot( np.array([ 0.5, -1, -u, 1.0/2.0 * u2]), Z )
        yr_pred[i] = np.dot( np.array([-0.5, -1, -u, 1.0/2.0 * u2]), Z )
    
    return (x_pred, yl_pred, yr_pred)



def LS_lane_residuals(lane_left, lane_right, Z):
    
    
    ## HIER CODE EINFUEGEN
    residual = np.zeros((lane_left.shape[0]+lane_right.shape[0], 1))
    for i in range(lane_left.shape[0]):
        u, v = lane_left[i,0], lane_left[i,1]
        u2 = u*u
        residual[i] = np.dot( 
            np.array([ 0.5, -1, -u, 1.0/2.0 * u2]), Z ) - v

    for i in range(lane_right.shape[0]):
        u, v = lane_right[i,0], lane_right[i,1]
        u2 = u*u
        u3 = u2*u 
        residual[lane_left.shape[0]+i] = np.dot( 
            np.array([-0.5, -1, -u, 1.0/2.0 * u2]), Z ) - v
    ## EIGENER CODE ENDE

    return residual

def LS_lane_inliers(residual, thresh):
    inlier= []
    ## HIER CODE EINFUEGEN#
    for i in range(residual.shape[0]):
      if abs(residual[i])<= thresh:
        inlier.append(residual [i]) 

    
    return np.array(inlier).shape[0] 

    ## EIGENER CODE ENDE
def Cauchy(r, sigma=1):
    """
    Cauchy loss function.
    
    Args:
        r: resiudals
        sigma: expected standard deviation of inliers
        
    Returns:
        w: vector of weight coefficients


    """

    ## HIER CODE EINFUEGEN
    c = 2.3849*sigma
    wi = np.zeros(len(r))
    for i in range(len(r)):
	wi[i] = 1/(1+np.power(r[i]/c, 2))
    return wi
    ## EIGENER CODE ENDE
    

def MEstimator_lane_fit(pL, pR, Z_initial, sigma=1, maxIteration=10):
    """
    M-Estimator for lane coeffients z=(W, Y_offset, Delta_Phi, c0)^T.
    
    Args:
        pL: [NL, 2]-array of left marking positions (in DIN70000) 
        pR: [NR, 2]-array of right marking positions (in DIN70000)
        Z_initial: the initial guess of the parameter vector
        sigma: the expecvted standard deviation of the inliers
        maxIteration: max number of iterations
    
    Returns:
        Z: lane coeffients (W, Y_offset, Delta_Phi, c0)
    """
    
    H = np.zeros((pL.shape[0]+pR.shape[0], 4)) # design matrix
    Y = np.zeros((pL.shape[0]+pR.shape[0], 1)) # noisy observations
    
    # fill H and Y for left line points
    for i in range(pL.shape[0]):
        u, v = pL[i,0], pL[i,1]
        u2 = u*u
        H[i, :] = [0.5, -1, -u, 1.0/2.0 * u2]
        Y[i] = v

    # fill H and Y for right line points
    for i in range(pR.shape[0]):
        u, v = pR[i,0], pR[i,1]
        u2 = u*u
        u3 = u2*u 
        H[pL.shape[0]+i, :] = [-0.5, -1, -u, 1.0/2.0 * u2]
        Y[pL.shape[0]+i] = v
        
    ## HIER CODE EINFUEGEN
    Z = Z_initial
    for i in range(maxIteration):

        # store old data
        Z0 = Z

        # compute residuals
        res = LS_lane_residuals(pL, pR, Z)

        # recompute weights
        W = np.diag(Cauchy(res, sigma))

        # recompute new estimate
        HTWH = np.dot(np.dot(H.T, W), H)
        inv_HTWH = np.linalg.inv(HTWH)
        Z = np.dot(np.dot(inv_HTWH, H.T), np.dot(W, Y))

    ## EIGENER CODE ENDE
    
    return Z

def plot_tangente(msg):
	
	## Laenge vom Fahrzeug (geschaetzt)
	L = 4

	## erhalte die Position vom Fahrzeug in x,y und yaw Koordinaten
	x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.orientation.z

	## berechne Position vom Front Achse

	fx = x + 0.5 * L * np.cos(np.arcsin(z))
    	fy = y + 0.5 * L * z

	# Search nearest point index
        dx_vec = fx - np.asarray(x_test).reshape([-1,1])
        dy_vec = fy - np.asarray(yR_test).reshape([-1,1])
        dist = np.hstack([dx_vec, dy_vec])
        dist_2 = np.sum(dist**2, axis=1)
        target_idx = np.argmin(dist_2)

	vor_target_idx = target_idx - 1

	## Tangente m = y1 - y0 / x1 - x0

	y1 = yR_test[target_idx]
	y0 = x_test[vor_target_idx]
	x1 = yR_test[target_idx]
	x0 = x_test[vor_target_idx]

	m = (y1 - y0) / (x1 - x0)
	
	print(m)


	
	
	
	


	
def plot_x(msg):

	global counter
	global x_test, yR_test
	## hier plotten wir die Strecke aus

	
	
	# M - Estimator

	M = np.vstack((msg.some_x, msg.some_y)).T
	max_range_m = 20
	roi_right_line = np.array([
	    [0, 0], 
	    [10, 0], 
	    [max_range_m, 5], 
	    [max_range_m, -13], 
	    [0, -13] ])

	roi_left_line = np.array([
	    [0, 0], 
	    [10, 0], 
	    [max_range_m, -5], 
	    [max_range_m, 13], 
	    [0, 13] ])

	lane_left = np.empty((0,2))
	lane_right = np.empty((0,2))

	for i in range(M.shape[0]):
	    if cv2.pointPolygonTest(roi_left_line, (M[i,0], M[i,1]), False) > 0:
		lane_left = np.vstack((lane_left, M[i,:])) 
	    if cv2.pointPolygonTest(roi_right_line, (M[i,0], M[i,1]), False) > 0:
		lane_right = np.vstack((lane_right, M[i,:])) 


	Z_initial = np.array([8, -4, np.radians(3.0), 0]).T 

	Z_MEst = MEstimator_lane_fit(lane_left, lane_right, Z_initial, 
                             sigma=0.2, maxIteration=10)
	x_pred, yl_pred, yr_pred = LS_lane_compute(Z_MEst)
	if counter % 3 == 0:
		plt.cla()
#		plt.plot(msg.some_y,msg.some_x,'.')
		plt.title('measurements')
		#plt.plot(lane_left[:, 1], lane_left[:, 0], 'r.')
		#plt.plot(lane_right[:, 1], lane_right[:, 0], 'g.')
		plt.plot(yl_pred, x_pred, 'b')
		plt.plot(yr_pred, x_pred, 'c')

		plt.xlabel('Y [m]')

		plt.ylabel('X [m]')
		plt.grid(True);
		plt.draw()
		plt.axis([-20,20, 0, 20])
		plt.pause(0.00000001)
		

	counter +=1 


	x_test = x_pred
	yR_test = yr_pred
	

if __name__ == '__main__':
    counter = 0
    x_test = []
    yR_test = []
    
    rospy.init_node('plot_node')
    rate = rospy.Rate(10.0)
    
    ## subscriben
    mysub = rospy.Subscriber('plot_topic', Foo, plot_x)
    carSub = rospy.Subscriber('base_pose_ground_truth', Odometry, plot_tangente)


    plt.ion()
    plt.show()
    rospy.spin()

