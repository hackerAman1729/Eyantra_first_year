#!/usr/bin/env python3

from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import time

class Swift:
    def _init_(self):
        rospy.init_node('drone_control')
        self.drone_position = [0.0, 0.0, 0.0]
        self.waypoints = [
            [0, 0, 23],
            [2, 0, 23],
            [2, 2, 23],
            [2, 2, 25],
            [-5, 2, 25],
            [-5, -3, 25],
            [-5, -3, 21],
            [7, -3, 21],
            [7, 0, 21],
            [0, 0, 19]
        ]
        self.current_waypoint_idx = 0
        self.alt_error = self.roll_error = self.pitch_error = 0.0
        self.prev_alt_error = self.prev_roll_error = self.prev_pitch_error = 0.0
        self.sum_alt_error = self.sum_roll_error = self.sum_pitch_error = 0.0

        self.thro_mul = [0, 0, 0]
        self.roll_mul = [0, 0, 0]
        self.pitch_mul = [0, 0, 0]

        self.init_command()
        self.init_ros_nodes()
        self.arm()

    def init_command(self):
        self.cmd = swift_msgs()
        self.cmd.rcRoll = self.cmd.rcPitch = self.cmd.rcYaw = self.cmd.rcThrottle = 1500
        self.cmd.rcAUX1 = self.cmd.rcAUX2 = self.cmd.rcAUX3 = self.cmd.rcAUX4 = 1500

    def init_ros_nodes(self):
        self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
        self.alt_error_pub = rospy.Publisher('/alt_error', Float64, queue_size=1)
        self.roll_error_pub = rospy.Publisher('/roll_error', Float64, queue_size=1)
        self.pitch_error_pub = rospy.Publisher('/pitch_error', Float64, queue_size=1)

        rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)

    def arm(self):
        self.disarm()
        self.init_command()
        self.cmd.rcThrottle = 1000
        self.cmd.rcAUX4 = 1500
        self.command_pub.publish(self.cmd)
        rospy.sleep(1)

    def disarm(self):
        self.cmd.rcAUX4 = 1100
        self.command_pub.publish(self.cmd)
        rospy.sleep(1)

    def whycon_callback(self, msg):
        self.drone_position = [msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z]

    # def pid(self):
    #     self.setpoint = self.waypoints[self.current_waypoint_idx]
    #     self.compute_error()
    #     self.update_command()
    #     self.command_pub.publish(self.cmd)

    def pid(self):
        self.setpoint = self.waypoints[self.current_waypoint_idx]
        if self.setpoint in [[-5, 2, 25], [-5, -3, 25], [-5, -3, 21]]:
            # Different PID gains
            # self.Kp = [0.09, 0.09, 0.05]
            # self.Ki = [0.0001, 0.0001, 0.0001]
            # self.Kd = [0.3, 0.3, 0.4]
            self.roll_mul = [0.09, 0.0001, 0.3]
            self.pitch_mul = [0.09, 0.0001, 0.3]
            self.thro_mul = [0.05, 0.0001, 0.4]            
        else:
            # Standard PID gains
            # self.Kp = [0.06, 0.06, 0.03]
            # self.Ki = [0.0001, 0.0001, 0.00005]
            # self.Kd = [0.3, 0.3, 0.3]
            self.roll_mul = [0.06, 0.0001, 0.3]
            self.pitch_mul = [0.06, 0.0001, 0.3]
            self.thro_mul = [0.03, 0.00005, 0.3]
        self.compute_error()
        self.update_command()
        self.command_pub.publish(self.cmd)

        # Check if the drone has reached the current waypoint
        if all(abs(self.drone_position[i] - self.setpoint[i]) < 0.2 for i in range(3)):
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1

    def compute_error(self):
        self.alt_error = self.drone_position[2] - self.setpoint[2]
        self.roll_error = self.drone_position[0] - self.setpoint[0]
        self.pitch_error = self.drone_position[1] - self.setpoint[1]

    def update_command(self):
        throttle = [2525, 2950, 3000]
        roll = [100, 0, 170]
        pitch = [135, 0, 650]

        # th_mul = [0.03, 0.0001, 0.3]
        # rp_mul = [0.06, 0.0001, 0.3]

        self.Kp = [roll[0]*self.roll_mul[0], pitch[0]*self.pitch_mul[0], throttle[0]*self.thro_mul[0]]
        self.Ki = [roll[1]*self.roll_mul[1], pitch[1]*self.pitch_mul[1], throttle[1]*self.thro_mul[1]]
        self.Kd = [roll[2]*self.roll_mul[2], pitch[2]*self.pitch_mul[2], throttle[2]*self.thro_mul[2]]

        self.cmd.rcThrottle = int(1500 + self.alt_error * self.Kp[2] +
                                   (self.alt_error - self.prev_alt_error) * self.Kd[2] +
                                   self.sum_alt_error * self.Ki[2])
        if self.cmd.rcThrottle > 2000: self.cmd.rcThrottle = 2000
        elif self.cmd.rcThrottle < 1000: self.cmd.rcThrottle = 1000
        self.prev_alt_error = self.alt_error
        self.sum_alt_error += self.alt_error
        self.alt_error_pub.publish(self.alt_error)

        self.cmd.rcRoll = int(1500 - self.roll_error * self.Kp[0] -
                               (self.roll_error - self.prev_roll_error) * self.Kd[0] -
                               self.sum_roll_error * self.Ki[0])
        if self.cmd.rcRoll > 2000: self.cmd.rcRoll = 2000
        elif self.cmd.rcRoll < 1000: self.cmd.rcRoll = 1000
        self.prev_roll_error = self.roll_error
        self.sum_roll_error += self.roll_error
        self.roll_error_pub.publish(self.roll_error)

        self.cmd.rcPitch = int(1500 + self.pitch_error * self.Kp[1] +
                                (self.pitch_error - self.prev_pitch_error) * self.Kd[1] +
                                self.sum_pitch_error * self.Ki[1])
        if self.cmd.rcPitch > 2000: self.cmd.rcPitch = 2000
        elif self.cmd.rcPitch < 1000: self.cmd.rcPitch = 1000
        self.prev_pitch_error = self.pitch_error
        self.sum_pitch_error += self.pitch_error
        self.pitch_error_pub.publish(self.pitch_error)


if _name_ == '_main_':
    swift_drone = Swift()
    r = rospy.Rate(30)
    rospy.sleep(8)

    while not rospy.is_shutdown():
        swift_drone.pid()
        r.sleep()