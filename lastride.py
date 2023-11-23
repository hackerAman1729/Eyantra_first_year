      #!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from luminosity_drone.msg import Biolocation
from swift_msgs.msg import *
import imutils

class SwiftDrone:
          def __init__(self):
              rospy.init_node('life_form_detector')
              self.bridge = CvBridge()
              self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
              self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
              self.biolocation_pub = rospy.Publisher('/astrobiolocation', Biolocation, queue_size=1)
              self.drone_position = [0.0, 0.0, 0.0]
              rospy.Subscriber('whycon/poses', PoseArray, self.whycon_callback)
              self.organism_detected = False
              self.roll_mul = self.pitch_mul = self.thro_mul = [1, 1, 1]  
              self.prev_alt_error = self.prev_roll_error = self.prev_pitch_error = 0.0

              self.cmd = swift_msgs()
              self.cmd.rcRoll = self.cmd.rcPitch = self.cmd.rcYaw = self.cmd.rcThrottle = 1500
              self.cmd.rcAUX1 = self.cmd.rcAUX2 = self.cmd.rcAUX3 = self.cmd.rcAUX4 = 1500
            

          def whycon_callback(self, msg):
              self.drone_position = [msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z]

          def disarm(self):
            self.cmd.rcAUX4 = 1100
            self.command_pub.publish(self.cmd)
            rospy.sleep(1)


          def image_callback(self, data):
                  try:
                    # Convert the ROS image to an OpenCV image
                    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

                    # Process the image to detect organisms
                    self.detect_organisms(cv_image)

                  except CvBridgeError as e:
                    rospy.logerr("CvBridge Error: {0}".format(e))

                  except Exception as ex:
                    rospy.logerr("Unexpected error during image processing: {0}".format(ex))

          def navigate_and_search(self):
            waypoints = [
                [-8, -8, 10],
                [-8, 8, 10],
                [8, 8, 10],
                [8, -8, 10]
            ]
            waypoint_index = 0
            while not rospy.is_shutdown() and waypoint_index < len(waypoints):
                target = waypoints[waypoint_index]
                self.setpoint = target
                while not rospy.is_shutdown():
                    self.compute_error()
                    self.update_command()
                    self.command_pub.publish(self.cmd)
                    if all(abs(self.drone_position[i] - self.setpoint[i]) < 1 for i in range(3)):
                        waypoint_index += 1
                        break
                    rospy.sleep(0.1)
                if self.organism_detected:
                    break


        def detect_organisms(self, image):
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          blurred = cv2.GaussianBlur(gray, (11, 11), 0)
          thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
          cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          cnts = imutils.grab_contours(cnts)
          num_leds = len(cnts)
            
          if num_leds >= 2 and not self.organism_detected:
            M = cv2.moments(cnts)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                self.organism_centroid = (cX, cY)
            else:
                self.organism_centroid = None

            self.organism_detected = True
            organism_type = f'alien_{chr(96 + num_leds)}'
            self.align_and_publish(organism_type)


          def align_and_publish(self, organism_type):
            if self.organism_centroid:
               
                frame_center_x, frame_center_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2

                while not rospy.is_shutdown():
                    aligned = self.adjust_drone_position_based_on_centroid(self.organism_centroid, frame_center_x, frame_center_y)
                    if aligned:
                        break
                    rospy.sleep(0.1)

            biolocation_msg = Biolocation()
            biolocation_msg.organism_type = organism_type
            biolocation_msg.whycon_x = self.drone_position[0]
            biolocation_msg.whycon_y = self.drone_position[1]
            biolocation_msg.whycon_z = self.drone_position[2]
            self.biolocation_pub.publish(biolocation_msg)

          def adjust_drone_position_based_on_centroid(self, centroid, frame_center_x, frame_center_y):
          cX, cY = centroid
          aligned = False

          # Tolerance for alignment, adjust as necessary
          tolerance = 10

          # Check if the centroid is approximately at the center of the frame
          if abs(cX - frame_center_x) < tolerance and abs(cY - frame_center_y) < tolerance:
              aligned = True
          else:
              # Adjust drone's position based on the centroid's position
              if cX < frame_center_x:
                  # Move drone left
                  self.cmd.rcRoll -= PID_ADJUSTMENT_VALUE
              elif cX > frame_center_x:
                  # Move drone right
                  self.cmd.rcRoll += PID_ADJUSTMENT_VALUE

              if cY < frame_center_y:
                  # Move drone down
                  self.cmd.rcPitch -= PID_ADJUSTMENT_VALUE
              elif cY > frame_center_y:
                  # Move drone up
                  self.cmd.rcPitch += PID_ADJUSTMENT_VALUE

              # Ensure commands are within valid range
              self.cmd.rcRoll = max(1000, min(2000, self.cmd.rcRoll))
              self.cmd.rcPitch = max(1000, min(2000, self.cmd.rcPitch))

              # Publish the new command
              self.command_pub.publish(self.cmd)

          return aligned



          def compute_error(self):
            self.alt_error = self.setpoint[2] - self.drone_position[2]
            self.roll_error = self.setpoint[0] - self.drone_position[0]
            self.pitch_error = self.setpoint[1] - self.drone_position[1]

            self.sum_alt_error += self.alt_error
            self.sum_roll_error += self.roll_error
            self.sum_pitch_error += self.pitch_error

          def update_command(self):
            roll = [100, 0, 170]  
            pitch = [135, 0, 650]  
            throttle = [2525, 2950, 3000]  

            self.Kp = [roll[0] * self.roll_mul[0], pitch[0] * self.pitch_mul[0], throttle[0] * self.thro_mul[0]]
            self.Ki = [roll[1] * self.roll_mul[1], pitch[1] * self.pitch_mul[1], throttle[1] * self.thro_mul[1]]
            self.Kd = [roll[2] * self.roll_mul[2], pitch[2] * self.pitch_mul[2], throttle[2] * self.thro_mul[2]]

            self.cmd.rcThrottle = int(1500 + self.alt_error * self.Kp[2] +
                                      (self.alt_error - self.prev_alt_error) * self.Kd[2] +
                                      self.sum_alt_error * self.Ki[2])
            self.cmd.rcRoll = int(1500 - self.roll_error * self.Kp[0] -
                                  (self.roll_error - self.prev_roll_error) * self.Kd[0] -
                                  self.sum_roll_error * self.Ki[0])
            self.cmd.rcPitch = int(1500 + self.pitch_error * self.Kp[1] +
                                   (self.pitch_error - self.prev_pitch_error) * self.Kd[1] +
                                   self.sum_pitch_error * self.Ki[1])

            self.cmd.rcThrottle = max(1000, min(2000, self.cmd.rcThrottle))
            self.cmd.rcRoll = max(1000, min(2000, self.cmd.rcRoll))
            self.cmd.rcPitch = max(1000, min(2000, self.cmd.rcPitch))

            self.prev_alt_error = self.alt_error
            self.prev_roll_error = self.roll_error
            self.prev_pitch_error = self.pitch_error



          def land_drone(self):
          self.setpoint = [11, 11, 37]
          while not rospy.is_shutdown():
              self.compute_error()
              self.update_command()
              self.command_pub.publish(self.cmd)
              if all(abs(self.drone_position[i] - self.setpoint[i]) < 0.2 for i in range(3)):
                  break
              rospy.sleep(0.1)
          self.disarm()


      if __name__ == '__main__':
          swift_drone = SwiftDrone()
          swift_drone.navigate_and_search()
          swift_drone.land_drone()
