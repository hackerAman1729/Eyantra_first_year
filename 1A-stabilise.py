#!/usr/bin/env python3

from swift_msgs.msg import *
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64
from pid_tune.msg import PidTune
import rospy
import time

class Swift:
    def __init__(self):
        rospy.init_node('drone_control')
        self.drone_position = [0.0, 0.0, 0.0]
        self.setpoint = [2, 2, 20]
        self.init_command()

        self.Kp = [0, 0, 0]
        self.Ki = [0, 0, 0]
        self.Kd = [0, 0, 0]

        self.alt_error = self.roll_error = self.pitch_error = 0.0
        self.prev_alt_error = self.prev_roll_error = self.prev_pitch_error = 0.0
        self.sum_alt_error = self.sum_roll_error = self.sum_pitch_error = 0.0

        self.min_throttle = self.min_roll = self.min_pitch = 1000
        self.max_throttle = self.max_roll = self.max_pitch = 2000

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
        # rospy.Subscriber('/pid_tuning_altitude', PidTune, self.altitude_set_pid)
        # rospy.Subscriber('/pid_tuning_roll', PidTune, self.roll_set_pid)
        # rospy.Subscriber('/pid_tuning_pitch', PidTune, self.pitch_set_pid)

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

    # def altitude_set_pid(self, alt):
    #     self.Kp[2] = alt.Kp * 0.03
    #     self.Ki[2] = alt.Ki * 0.00005
    #     self.Kd[2] = alt.Kd * 0.3

    # def roll_set_pid(self, roll):
    #     self.Kp[0] = roll.Kp * 0.06
    #     self.Ki[0] = roll.Ki * 0.0001
    #     self.Kd[0] = roll.Kd * 0.3

    # def pitch_set_pid(self, pitch):
    #     self.Kp[1] = pitch.Kp * 0.06
    #     self.Ki[1] = pitch.Ki * 0.0001
    #     self.Kd[1] = pitch.Kd * 0.3

    def pid(self):
        self.compute_error()
        self.update_command()
        self.command_pub.publish(self.cmd)

    def compute_error(self):
        self.alt_error = self.drone_position[2] - self.setpoint[2]
        self.roll_error = self.drone_position[0] - self.setpoint[0]
        self.pitch_error = self.drone_position[1] - self.setpoint[1]

    def update_command(self):
        throttle = [2475, 2950, 2975]
        roll = [100, 0, 170]
        pitch = [135, 0, 650]

        th_mul = [0.03, 0.0001, 0.3]
        rp_mul = [0.06, 0.0001, 0.3]

        self.Kp = [roll[0]*rp_mul[0], pitch[0]*rp_mul[0], throttle[0]*th_mul[0]]
        self.Ki = [roll[1]*rp_mul[1], pitch[1]*rp_mul[1], throttle[1]*th_mul[1]]
        self.Kd = [roll[2]*rp_mul[2], pitch[2]*rp_mul[2], throttle[2]*th_mul[2]]

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

if __name__ == '__main__':
    swift_drone = Swift()
    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        swift_drone.pid()
        r.sleep()

