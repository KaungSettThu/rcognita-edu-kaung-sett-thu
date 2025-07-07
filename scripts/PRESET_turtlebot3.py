"""
Preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator).

"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import rospy
  
import pathlib  
  
import warnings
import csv
from datetime import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import math

import turtlebot3_rcognita.systems as systems
import turtlebot3_rcognita.controllers as controllers
import turtlebot3_rcognita.models as models
import turtlebot3_rcognita.loggers as loggers
import turtlebot3_rcognita.visuals as visuals
import turtlebot3_rcognita.simulator as simulator
from turtlebot3_rcognita.utilities import on_key_press

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

from math import atan2, sin, cos, sqrt, pi


import argparse

try:
    rospy.init_node('turtlebot3_rcognita', anonymous = True)
except rospy.ROSException as e:
    rospy.logwarn(f"Node initialization failed: {str(e)}")


#----------------------------------------Set up dimensions
dim_state = 3
dim_input = 2
dim_output = dim_state
dim_disturb = 0

dim_R1 = dim_output + dim_input
dim_R2 = dim_R1

description = "Agent-environment preset: a 3-wheel robot (kinematic model a.k.a. non-holonomic integrator)."

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('--ctrl_mode', metavar='ctrl_mode', type=str,
                    choices=['MPC',
                             "N_CTRL",
                             'LQR'],
                    default='N_CTRL',
                    help='Control mode. Currently available: ' +
                    '----manual: manual constant control specified by action_manual; ' +
                    '----nominal: nominal controller, usually used to benchmark optimal controllers;' +                     
                    '----MPC:model-predictive control; ' +
                    '----RQL: Q-learning actor-critic with Nactor-1 roll-outs of running objective; ' +
                    '----SQL: stacked Q-learning; ' + 
                    '----RLStabLyap: (experimental!) learning agent with Lyapunov-like stabilizing contraints.')
parser.add_argument('--dt', type=float, metavar='dt',
                    default=0.1,
                    help='Controller sampling time.' )
parser.add_argument('--t1', type=float, metavar='t1',
                    default=30,
                    help='Final time of episode.' )
parser.add_argument('--Nruns', type=int,
                    default=1,
                    help='Number of episodes. Learned parameters are not reset after an episode.')
parser.add_argument('--is_log_data', type=int,
                    default=1,
                    help='Flag to log data into a data file. Data are stored in simdata folder.')
parser.add_argument('--is_visualization', type=int,
                    default=1,
                    help='Flag to produce graphical output.')
parser.add_argument('--is_print_sim_step', type=int,
                    default=1,
                    help='Flag to print simulation data into terminal.')
parser.add_argument('--action_manual', type=float,
                    default=[-5, -3], nargs='+',
                    help='Manual control action to be fed constant, system-specific!')
parser.add_argument('--Nactor', type=int,
                    default=6,
                    help='Horizon length (in steps) for predictive controllers.')
parser.add_argument('--pred_step_size_multiplier', type=float,
                    default=5.0,
                    help='Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.')
parser.add_argument('--buffer_size', type=int,
                    default=25,
                    help='Size of the buffer (experience replay) for model estimation, agent learning etc.')
parser.add_argument('--run_obj_struct', type=str,
                    default='quadratic',
                    choices=['quadratic',
                             'biquadratic'],
                    help='Structure of running objective function.')
parser.add_argument('--R1_diag', type=float, nargs='+',
                #     Stable LQR R1
                    default=[2.5, 2.5, 20, 3, 2], 
                #     Stable MPC R1
                #     default=[5, 30, 2, 2, 3],
                    help='Parameter of running objective function. Must have proper dimension. ' +
                    'Say, if chi = [observation, action], then a quadratic running objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--R2_diag', type=float, nargs='+',
                    default=[1, 10, 1, 0, 0],
                    help='Parameter of running objective function . Must have proper dimension. ' + 
                    'Say, if chi = [observation, action], then a bi-quadratic running objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, ' +
                    'where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--Ncritic', type=int,
                    default=25,
                    help='Critic stack size (number of temporal difference terms in critic cost).')
parser.add_argument('--gamma', type=float,
                    default=0.9,
                    help='Discount factor.')
parser.add_argument('--critic_period_multiplier', type=float,
                    default=1.0,
                    help='Critic is updated every critic_period_multiplier times dt seconds.')
parser.add_argument('--critic_struct', type=str,
                    default='quad-mix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix',
                                                   'quad-mix',
                                                   'poly3',
                                                   'poly4'],
                    help='Feature structure (critic). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms; ' +
                    '----quad-mix: quadratic, mixed observation-action terms (for, say, Q or advantage function approximations); ' +
                    '----poly3: 3-order model, see the code for the exact structure; ' +
                    '----poly4: 4-order model, see the code for the exact structure. '
                    )
parser.add_argument('--actor_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix'],
                    help='Feature structure (actor). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms.')
parser.add_argument('--init_robot_pose_x', type=float,
                    default=-3.0,
                    help='Initial x-coordinate of the robot pose.')
parser.add_argument('--init_robot_pose_y', type=float,
                    default=-3.0,
                    help='Initial y-coordinate of the robot pose.')
parser.add_argument('--init_robot_pose_theta', type=float,
                    default=0.0,
                    help='Initial orientation angle (in radians) of the robot pose.')

# Goal coordinates
parser.add_argument('--goal_robot_pose_x', type=float,
                    default=0.0,
                    help='Goal x-coordiante of the robot pose.')
parser.add_argument('--goal_robot_pose_y', type=float,
                    default=0.0,
                    help='Goal y-coordiante of the robot pose.')
parser.add_argument('--goal_robot_pose_theta', type=float,
                    default=1.57,
                    help='Goal orientation angle (in radians) of the robot pose.')

parser.add_argument('--distortion_x', type=float,
                    default=-0.6,
                    help='X-coordinate of the center of distortion.')
parser.add_argument('--distortion_y', type=float,
                    default=-0.5,
                    help='Y-coordinate of the center of distortion.')
parser.add_argument('--distortion_sigma', type=float,
                    default=0.1,
                    help='Standard deviation (sigma) of distortion.')
parser.add_argument('--seed', type=int,
                    default=1,
                    help='Seed for random number generation.')

# args = parser.parse_args()

# Parse known arguments
args, unknown = parser.parse_known_args()


seed=args.seed
print(seed)

xdistortion_x = args.distortion_x
ydistortion_y = args.distortion_y
distortion_sigma = args.distortion_sigma

x = args.init_robot_pose_x
y = args.init_robot_pose_y
theta = args.init_robot_pose_theta

while theta > np.pi:
        theta -= 2 * np.pi
while theta < -np.pi:
        theta += 2 * np.pi

state_init = np.array([x, y, theta])

# setting the goal coordinates

x_goal = args.goal_robot_pose_x
y_goal = args.goal_robot_pose_y
theta_goal = args.goal_robot_pose_theta

while theta_goal > np.pi:
        theta_goal -= 2 * np.pi
while theta_goal < -np.pi:
        theta_goal += 2 * np.pi

state_goal = np.array([x_goal, y_goal, theta_goal])

args.action_manual = np.array(args.action_manual)

pred_step_size = args.dt * args.pred_step_size_multiplier
critic_period = args.dt * args.critic_period_multiplier

R1 = np.diag(np.array(args.R1_diag))
R2 = np.diag(np.array(args.R2_diag))

assert args.t1 > args.dt > 0.0
assert state_init.size == dim_state

globals().update(vars(args))

#----------------------------------------Fixed settings
is_disturb = 0
is_dyn_ctrl = 0

t0 = 0

action_init = 0 * np.ones(dim_input)

# Solver
atol = 1e-3
rtol = 1e-2

# xy-plane
xMin = -4#-1.2
xMax = 0.2
yMin = -4#-1.2
yMax = 0.2

# xy-offsets
# to ensure that the turtlebot starts from non-zero condition
x_offset = 3
y_offset = 3

# Control constraints
v_min = -0.22
v_max = 0.22
omega_min = -2.84
omega_max = 2.84

ctrl_bnds=np.array([[v_min, v_max], [omega_min, omega_max]])

#----------------------------------------Initialization : : system
my_sys = systems.Sys3WRobotNI(sys_type="diff_eqn", 
                                     dim_state=dim_state,
                                     dim_input=dim_input,
                                     dim_output=dim_output,
                                     dim_disturb=dim_disturb,
                                     pars=[],
                                     ctrl_bnds=ctrl_bnds,
                                     is_dyn_ctrl=is_dyn_ctrl,
                                     is_disturb=is_disturb,
                                     pars_disturb=[])

observation_init = my_sys.out(state_init)

xCoord0 = state_init[0]
yCoord0 = state_init[1]
alpha0 = state_init[2]
alpha_deg_0 = alpha0/2/np.pi


#----------------------------------------Initialization : : model

#----------------------------------------Initialization : : controller
my_ctrl_nominal = controllers.N_CTRL(observation_target = state_goal,
                                        ctrl_bnds = ctrl_bnds) 

my_ctrl_lqr = controllers.LQR(dim_state,
                                dim_input,
                                dim_output,
                                sampling_time = dt, 
                                system = my_sys,
                                run_obj_pars = np.array(R1),
                                observation_target = state_goal,
                                state_init = state_init,
                                ctrl_bnds = ctrl_bnds)

# Predictive optimal controller
my_ctrl_opt_pred = controllers.ControllerOptimalPredictive(dim_state,
                                            dim_input,
                                            dim_output,
                                            ctrl_mode,
                                            system = my_sys,
                                            ctrl_bnds = ctrl_bnds,
                                            action_init = [],
                                            t0 = t0,
                                            sampling_time = dt,
                                            Nactor = Nactor,
                                            pred_step_size = pred_step_size,
                                            sys_rhs = my_sys._state_dyn,
                                            sys_out = my_sys.out,
                                            state_sys = state_init,
                                            buffer_size = buffer_size,
                                            gamma = gamma,
                                            Ncritic = Ncritic,
                                            critic_period = critic_period,
                                            critic_struct = critic_struct,
                                            run_obj_struct = run_obj_struct,
                                            run_obj_pars = np.array(R1),
                                            observation_target = state_goal,
                                            state_init = state_init,
                                            obstacle = [xdistortion_x, ydistortion_y,distortion_sigma],
                                            seed = seed)


my_ctrl_benchm = my_ctrl_opt_pred

'''
    Integration of turtlebot3 with the packages from rcognita
    
    - odom_callback() is used to take the Odometry measurements form Trutlebot3

    - run() is used to return the Twist commands to the Turtlebot3
'''

print("Running Turtlebot3 with the use of rcognita-edu")

"""
Turtlbe bot class to take odem readings from the bot
and publish twist commands
"""

def odom_callback(msg):
        """
        Handler method to take odom readings
        """
        global t, observation 

        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation

        _, _, theta = euler_from_quaternion([
                orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        ])
        
        # offset to make the turtlebot start at a non-zero position

        observation = np.array([position.x, position.y, theta])
        observation = observation - np.array([x_offset, y_offset, 0])

        t = msg.header.stamp.to_sec()


rospy.loginfo("Turtlebot 3 Sigwart et. al. Controller Started...")

# substribe to odom
rospy.Subscriber('/odom', Odometry, odom_callback)

# Publisher object for publishing twist
cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

rate = rospy.Rate(10)

t = 0.0
observation = observation_init

while not rospy.is_shutdown():

        action = np.array(controllers.ctrl_selector(t, observation, action_manual, my_ctrl_nominal, my_ctrl_benchm, ctrl_mode))
        
        
        # create a tiwst object

        twist = Twist()

        # set the linear velocities

        twist.linear.x = action[0]
        twist.linear.y = 0
        twist.linear.z = 0

        # set the angular velocities

        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = action[1]

        # publich the twist

        cmd_pub.publish(twist)


        my_ctrl_benchm.receive_sys_state(my_sys._state)
        my_ctrl_benchm.upd_accum_obj(observation, action)
        
        xCoord = observation[0]
        yCoord = observation[1]
        alpha = observation[2]

        linear_vel = action[0]
        angular_vel = action[1]

        # Debugging
        print(f"x: {xCoord} y: {yCoord} theta: {alpha}")
        print(f"action - v: {linear_vel} w: {angular_vel}")
        print("")
        
        run_obj = my_ctrl_benchm.run_obj(observation, action)
        accum_obj = my_ctrl_benchm.accum_obj_val

        rospy.Rate(10).sleep()
