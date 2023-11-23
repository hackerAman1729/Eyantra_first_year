      #!/usr/bin/env python3

      from swift_msgs.msg import *
      from geometry_msgs.msg import PoseArray
      from std_msgs.msg import Float64
      from pid_tune.msg import PidTune
      import rospy
      import time
      import numpy as np
      from sensor_msgs.msg import Image
      from cv_bridge import CvBridge, CvBridgeError
      import cv2
      import logging
      from luminosity_drone.msg import Biolocation

      logging.basicConfig(filename='ldlog.txt', level=logging.INFO, 
                          format='%(asctime)s:%(levelname)s:%(message)s')

      class Swift:
          def _init_(self):
              rospy.init_node('drone_control')
              self.drone_position = [0.0, 0.0, 0.0]
              self.landing = False
              self.final_movement_initiated = False

              self.bridge = CvBridge()
              self.image_sub = rospy.Subscriber('/swift/camera_rgb/image_raw', Image, self.image_callback)
              self.led_detection_active = False
              self.landing = False
              self.final_movement_initiated = False


              self.arena_bounds = [(-8, -8), (8, -8), (8, 8), (-8, 8)]
              self.waypoints = self.generate_arena_waypoints()
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

          def generate_arena_waypoints(self):
              grid_spacing = 4  
              x_coords = np.arange(self.arena_bounds[0][0], self.arena_bounds[1][0] + grid_spacing, grid_spacing)
              y_coords = np.arange(self.arena_bounds[0][1], self.arena_bounds[2][1] + grid_spacing, grid_spacing)

              waypoints = []
              for x in x_coords:
                  for y in y_coords:
                      waypoints.append([x, y, 23]) 
              return waypoints

          def init_command(self):
              self.cmd = swift_msgs()
              self.cmd.rcRoll = self.cmd.rcPitch = self.cmd.rcYaw = self.cmd.rcThrottle = 1500
              self.cmd.rcAUX1 = self.cmd.rcAUX2 = self.cmd.rcAUX3 = self.cmd.rcAUX4 = 1500

          def init_ros_nodes(self):
              self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
              self.alt_error_pub = rospy.Publisher('/alt_error', Float64, queue_size=1)
              self.roll_error_pub = rospy.Publisher('/roll_error', Float64, queue_size=1)
              self.pitch_error_pub = rospy.Publisher('/pitch_error', Float64, queue_size=1)
              self.bio_location_pub = rospy.Publisher('astrobiolocation', Biolocation, queue_size=1)


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

          def image_callback(self, msg):
         
            print("Images are being recieved")
            try:
                cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                print("cv2_img conversion successful")
            except CvBridgeError as e:
                logging.error("CvBridge Error: {0}".format(e))  
            else:
                if self.led_detection_active:
                    print("Detect leds running")
                    self.detect_leds(cv2_img)


          def detect_leds(self, frame):
            print("Processing frame for LED detection")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            led_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]

            if led_contours:
                  M = cv2.moments(np.concatenate(led_contours))
                  if M["m00"] != 0:
                      centroid_x = int(M["m10"] / M["m00"])
                      centroid_y = int(M["m01"] / M["m00"])

                      center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2

                      offset_x = centroid_x - center_x
                      offset_y = centroid_y - center_y

                      
                      self.align_drone(offset_x, offset_y)
                      self.publish_biolocation(centroid_x, centroid_y, self.drone_position)
                  else:
                      logging.info("No LED contours detected.")
                  self.final_movement_initiated = False 
            print(led_contours)
            if len(led_contours) == 3:        
              print("LED found")

          def publish_biolocation(self, led_x, led_y, whycon_pos):
              bio_loc_msg = Biolocation()
              bio_loc_msg.organism_type = "alien_b"  
              bio_loc_msg.whycon_x = whycon_pos[0]
              bio_loc_msg.whycon_y = whycon_pos[1]
              bio_loc_msg.whycon_z = whycon_pos[2]

              print(f"Published Biolocation: organism_type={bio_loc_msg.organism_type}, whycon_x={bio_loc_msg.whycon_x}, whycon_y={bio_loc_msg.whycon_y}, whycon_z={bio_loc_msg.whycon_z}")

              self.bio_location_pub.publish(bio_loc_msg)
              logging.info(f"Published Biolocation: {bio_loc_msg}")

              if not self.final_movement_initiated:
                  self.initiate_final_movement()

          def initiate_final_movement(self):
              self.final_movement_initiated = True
              self.waypoints.append([11, 11, 37])  
              self.current_waypoint_idx = len(self.waypoints) - 1 
          def align_drone(self, offset_x, offset_y):
            
            sensitivity = 0.1
            roll_adjust = -offset_x * sensitivity
            pitch_adjust = -offset_y * sensitivity

            
            new_roll = max(min(self.cmd.rcRoll + roll_adjust, 2000), 1000)
            new_pitch = max(min(self.cmd.rcPitch + pitch_adjust, 2000), 1000)

            self.cmd.rcRoll = int(new_roll)
            self.cmd.rcPitch = int(new_pitch)

            self.command_pub.publish(self.cmd)

            print(f"Aligning drone: roll={new_roll}, pitch={new_pitch}")

          def whycon_callback(self, msg):
              self.drone_position = [msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z]

          

          def pid(self):
              self.setpoint = self.waypoints[self.current_waypoint_idx]
              if self.setpoint in [[-5, 2, 25], [-5, -3, 25], [-5, -3, 21]]:
                 
                  self.roll_mul = [0.09, 0.0001, 0.3]
                  self.pitch_mul = [0.09, 0.0001, 0.3]
                  self.thro_mul = [0.05, 0.0001, 0.4]            
              else:
                  
                  self.roll_mul = [0.06, 0.0001, 0.3]
                  self.pitch_mul = [0.06, 0.0001, 0.3]
                  self.thro_mul = [0.03, 0.00005, 0.3]
              self.compute_error()
              self.update_command()
              self.command_pub.publish(self.cmd)
              self.led_detection_active = True


                        if all(abs(self.drone_position[i] - self.setpoint[i]) < 0.2 for i in range(3)):
                          if self.current_waypoint_idx < len(self.waypoints) - 1:
                              self.current_waypoint_idx += 1
                          else:
                              
                              if not self.landing:
                                  print("Reached final waypoint. Preparing to land.")
                                  self.landing = True
                                  self.land()

          def land(self):
          
              if self.landing:
                  return
              self.landing = True
              print("Landing sequence initiated.")

             
              while not rospy.is_shutdown() and self.drone_position[2] > 0.5: 
                  self.decrease_altitude()
                  rospy.sleep(0.1)  
              print("Drone is landing.")
              self.disarm()
              print("Drone has landed and disarmed.")


          def decrease_altitude(self):
          
          self.cmd.rcThrottle -= 10  
          if self.cmd.rcThrottle < 1000:  
              self.cmd.rcThrottle = 1000
          self.command_pub.publish(self.cmd)  
          self.alt_error_pub.publish(self.alt_error)  



          def compute_error(self):
              self.alt_error = self.drone_position[2] - self.setpoint[2]
              self.roll_error = self.drone_position[0] - self.setpoint[0]
              self.pitch_error = self.drone_position[1] - self.setpoint[1]

          def update_command(self):
              throttle = [2525, 2950, 3000]
              roll = [100, 0, 170]
              pitch = [135, 0, 650]

             

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
          rospy.loginfo("Starting Swift drone node")  
          print("Swift drone launched")

          swift_drone = Swift()
          r = rospy.Rate(30)
          swift_drone.waypoints.append([11, 11, 37])  

          while not rospy.is_shutdown():
              swift_drone.pid()
              r.sleep()