#!/usr/bin/env python 

from __future__ import print_function
import rospy
import tf
from car_demo.msg import Control, Trajectory, ControlDebug, LaneXandY
import numpy as np
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry

import message_filters
import std_msgs.msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
from simulation_image_helper import SimulationImageHelper
#from rospy.numpy_msg import numpy_msg

from car_demo.msg import LaneXandY

## Funktion fuer die Langsregelung ##
def speed_control(target, current):
	    """
	    Proportional control for the speed.
	    :param target: target speed (m/s)
	    :param current: current speed (m/s)
	    :return: controller output (m/ss)
	    """
	    ## INSERT CODE HERE
	    return Kp * (target - current)
	    ## END INSERTED CODE

## Funktion um den nearest Point zu berechnen ##
def calc_target_index(state, cx, cy, cyaw):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    : state[0] = x, state[1] = y, state[2] = v, state[3] = yaw
    :param cx: [m] x-coordinates of (sampled) desired trajectory
    :param cy: [m] y-coordinates of (sampled) desired trajectory
    :param cyaw: [rad] tangent angle of (sampled) desired trajectory
    :return: (int, float)
    """
    # Calc front axle position
    fx = state[0] + 0.5 * L * np.cos(state[3])
    fy = state[1] + 0.5 * L * np.sin(state[3])

    # Search nearest point index
    dx_vec = fx - np.asarray(cx).reshape([-1,1])
    dy_vec = fy - np.asarray(cy).reshape([-1,1])
    dist = np.hstack([dx_vec, dy_vec])
    dist_2 = np.sum(dist**2, axis=1)
    target_idx = np.argmin(dist_2)

    # Project RMS error onto front axle vector
    front_axle_vec = [np.cos(cyaw[target_idx] + np.pi / 2),
                      np.sin(cyaw[target_idx] + np.pi / 2)]
    error_front_axle = np.dot(dist[target_idx,:], front_axle_vec)

    return target_idx, error_front_axle

## STANLEY-REGLER - FUNKTION ##
def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: [m] x-coordinates of (sampled) desired trajectory
    :param cy: [m] y-coordinates of (sampled) desired trajectory
    :param cyaw: [rad] orientation of (sampled) desired trajectory
    :param last_target_idx: [int] last visited point on desired trajectory
    :return: ([rad] steering angle, 
        [int] last visited point on desired trajectory, 
        [m] cross track error at front axle)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy, cyaw)

    # make sure that we never match a point on the desired path 
    # that we already passed earlier:
    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    ## INSERT CODE HERE
    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state[3])
    # theta_d corrects the cross track error
    theta_d = np.arctan2(-k * error_front_axle, state[2])
    # Steering control
    delta = theta_e + theta_d
    ## END INSERTED CODE

    return delta, current_target_idx, error_front_axle

## erhalte die x_pred und yR_pred Werte von plot_node.py ##
def getXandY(msg):
    pass
    
    #test1 = msg.x_pred
    #test2 = msg.yR_pred
    #return test1,test2


if __name__ == '__main__':
    rospy.init_node('test1_node')
    rate = rospy.Rate(10.0)

    v_desired = 5 # Sollgeschwindigkeit = 5 km/h
    Kp = 1.0  # speed propotional gain
    L = 4 # Laenge vom Fahrzeug
    k = 0.7  # control gain

    # subscribers
    listener = tf.TransformListener()
    jointStateSub = message_filters.Subscriber("joint_states", JointState)
    jointStateCache = message_filters.Cache(jointStateSub, 100)
	
    ## Hier wird x_pred und yR_pred von plot_node.py subscribed ##
    mysub = rospy.Subscriber('XandY_topic', LaneXandY, getXandY)
	
    ## Hier subscriben wir die Position x,y und den Yaw vom Fahrzeug
    carPositionSub = message_filters.Subscriber("base_pose_ground_truth", Odometry)
    carPositionCache = message_filters.Cache(carPositionSub, 100)

    # publishers
    pubControl = rospy.Publisher('prius', Control, queue_size=1)
    
    # main loop
    time_start = rospy.Time(0)
    while not rospy.is_shutdown():
        time_now = rospy.get_rostime()
        if time_start == rospy.Time(0):
            time_start = time_now

        # get vehicle position data 
        try:
            (trans, rot) = listener.lookupTransform('/map', '/din70000', rospy.Time(0)) # get latest trafo between world and vehicle
            rpy = tf.transformations.euler_from_quaternion(rot)
            #rospy.loginfo("pos: T=%s, rpy=%s" % (str(trans), str(rpy)) )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #rospy.loginfo("Could not receive position information!")
            rate.sleep()
            continue
	
        
        # get vehicle joint data
        joint_state = jointStateCache.getElemBeforeTime(time_now)
        joint_state_dict = dict(zip(joint_state.name, joint_state.velocity))
        v_kmh = 0.25*(joint_state_dict['front_right_wheel_joint'] + \
            joint_state_dict['rear_left_wheel_joint'] + \
            joint_state_dict['front_left_wheel_joint'] + \
            joint_state_dict['rear_right_wheel_joint'])
        #rospy.loginfo("v_kmh = " + str(v_kmh))

	#Laengsregelung (wenn die Beschleunigung groesser wie 0 ist, soll es mit 0.1 throttlen und wenn nicht dann leicht abbremsen) ##
	for i in range(100):
	    acc = speed_control(v_desired, v_kmh)
	    if (acc > 0):
		throttle_desired = 0.1
	    else:
		throttle_desired = -0.05

	## Stanley-Regler
	
	## Hier erhalten wir die x,y und yaw vom Fahrzeug

	model_states = carPositionCache.getElemBeforeTime(time_now)
	x1 = model_states.pose.pose.position.x
	y1 = model_states.pose.pose.position.y
	yaw1 = model_states.twist.twist.angular.z
        #rospy.loginfo("x-Position = " + str(x))
	#rospy.loginfo("y-Position = " + str(y))
	#rospy.loginfo("yaw-Position = " + str(yaw))


	target_speed = v_desired

	# Hier initalisieren wir die Anfangswerte vom Fahrzeug
	state = []
	x_init = -0.0
	y_init = 5.0
	yaw_init = np.radians(20.0)
	v_init = 0.0
	state.append(x_init)
	state.append(y_init)
	state.append(v_init)
	state.append(yaw_init)

	## Stanley-Regler
	
	state.append(x1)
	state.append(y1)
	state.append(v_kmh)
	state.append(yaw1)

	x = state[0]
	y = state[1]
	v = state[2]
	yaw = state[3]
	
	#xTest, yRTest = mysub
	
	
	
	#e_track = [np.nan]
	target_idx, _ = calc_target_index(state, xTest, yRTest, state[3])
 	## INSERT CODE HERE
	di, target_idx, dlat = stanley_control(state, xTest, yRTest, state[3], target_idx)
        ## END INSERTED CODE

	print(di)
		
	#di # Lenkwinkel in [rad]
	#dlat # Error (in Meter) von der Fahrbahn zum Fahrzeug Front-Achse




        # send driving commands to vehicle
        command = Control()
        command.header = std_msgs.msg.Header()
        command.header.stamp = time_now
        command.shift_gears = Control.FORWARD
        if throttle_desired > 0:
            command.throttle, command.brake = np.clip(throttle_desired, 0, 1), 0
        else:
            command.throttle, command.brake = 0, np.clip(-throttle_desired, 0, 1)
        #command.steer = 0.9
        pubControl.publish(command)

        rate.sleep()








