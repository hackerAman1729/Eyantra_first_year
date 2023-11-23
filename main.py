
    from sensor_msgs.msg import Image
    from luminosity_drone.msg import Biolocation
    from geometry_msgs.msg import PoseArray
    from std_msgs.msg import Float64
    from swift_msgs.msg import swift_msgs
    from cv_bridge import CvBridge
    import rospy
    import cv2
    import numpy as np
    from skimage import measure
    from imutils import contours
    import imutils

    class SwiftDrone:
        def __init__(self):
            rospy.init_node('life_form_detector', anonymous=True)
            self.image_sub = rospy.Subscriber("/drone_camera/image_raw", Image, self.image_callback)
            self.pose_sub = rospy.Subscriber('whycon/poses', PoseArray, self.pose_callback)
            self.location_pub = rospy.Publisher('/astrobiolocation', Biolocation, queue_size=10)
            self.command_pub = rospy.Publisher('/drone_command', swift_msgs, queue_size=1)
            self.cv_bridge = CvBridge()
            self.current_pose = None
            self.found_organisms = set()
            self.search_grid = self.generate_search_grid()
            self.waypoint_index = 0
            self.init_command()
            self.waypoints =  [
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

            self.Kp = [0.06, 0.06, 0.03]  
            self.Ki = [0.0001, 0.0001, 0.0001]  
            self.Kd = [0.3, 0.3, 0.3] 

        def generate_search_grid(self):
          grid = []
          x_min, x_max = -8, 8
          y_min, y_max = -8, 8
          
          step_size = 1.0

          y = y_min
          while y <= y_max:
              x = x_min if (y / step_size) % 2 == 0 else x_max
              while 0 <= x <= x_max if (y / step_size) % 2 == 0 else x_max >= x >= x_min:
                  grid.append((x, y, 20))  
                  x += step_size if (y / step_size) % 2 == 0 else -step_size
              y += step_size

          return grid


        def init_command(self):
          self.cmd = swift_msgs()
          self.cmd.rcRoll = 1500
          self.cmd.rcPitch = 1500
          self.cmd.rcYaw = 1500
          self.cmd.rcThrottle = 1500
          
          self.cmd.rcAUX1 = 1500
          self.cmd.rcAUX2 = 1500
          self.cmd.rcAUX3 = 1500
          self.cmd.rcAUX4 = 1500



        def arm(self):
         
          self.cmd.rcThrottle = 1000
          self.cmd.rcAUX4 = 2000  
          self.command_pub.publish(self.cmd)
          rospy.sleep(1) 
          self.cmd.rcRoll = 1500
          self.cmd.rcPitch = 1500
          self.cmd.rcYaw = 1500
          self.cmd.rcAUX1 = 1500
          self.cmd.rcAUX2 = 1500
          self.cmd.rcAUX3 = 1500
          self.cmd.rcAUX4 = 1500
          self.command_pub.publish(self.cmd)
          rospy.sleep(1)



        def disarm(self):
          
          self.cmd.rcAUX4 = 1000
          
          self.cmd.rcThrottle = 1000
          self.command_pub.publish(self.cmd)
          rospy.sleep(1)  

          self.cmd.rcRoll = 1500
          self.cmd.rcPitch = 1500
          self.cmd.rcYaw = 1500
          self.cmd.rcAUX1 = 1500
          self.cmd.rcAUX2 = 1500
          self.cmd.rcAUX3 = 1500
          self.command_pub.publish(self.cmd)
          rospy.sleep(1)



        def pose_callback(self, msg):
       
         if msg.poses:
            pose = msg.poses[0]
            self.current_pose = (pose.position.x, pose.position.y, pose.position.z)


      def image_callback(self, data):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        labels = measure.label(thresh, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")

        for label in np.unique(labels):
            if label == 0:
                continue

            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            if numPixels > 300: 
                mask = cv2.add(mask, labelMask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        led_count = 0
        for c in cnts:
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:  
                led_count += 1

        if led_count > 0:
            organism_type = ''
            if led_count == 2:
                organism_type = 'alien_a'
            elif led_count == 3:
                organism_type = 'alien_b'
            elif led_count == 4:
                organism_type = 'alien_c'
            else:
                rospy.loginfo('Unknown organism type detected')
                return  

            if self.current_pose and organism_type:
                self.align_and_publish(organism_type, self.current_pose)

        center_offset_x = cv_image.shape[1]//2 - cX
        center_offset_y = cv_image.shape[0]//2 - cY

        rospy.loginfo(f"Offset from center: x={center_offset_x}, y={center_offset_y}")



        def navigate_to_waypoint(self):
          if self.current_pose is None or self.waypoint_index >= len(self.search_grid):
            return False

          waypoint = self.search_grid[self.waypoint_index]
          dx = waypoint[0] - self.current_pose[0]
          dy = waypoint[1] - self.current_pose[1]
          dz = waypoint[2] - self.current_pose[2]

          threshold = 0.5  
          if abs(dx) < threshold and abs(dy) < threshold and abs(dz) < threshold:
            self.waypoint_index += 1
            if self.waypoint_index >= len(self.search_grid):
                return True  
            else:
                return False  
          Kp = 0.5
          self.cmd.rcPitch = int(1500 + Kp * dy)
          self.cmd.rcRoll = int(1500 - Kp * dx)
          self.cmd.rcThrottle = int(1500 + Kp * dz)

          self.cmd.rcPitch = max(1000, min(self.cmd.rcPitch, 2000))
          self.cmd.rcRoll = max(1000, min(self.cmd.rcRoll, 2000))
          self.cmd.rcThrottle = max(1000, min(self.cmd.rcThrottle, 2000))

          self.command_pub.publish(self.cmd)
          return False



        def align_and_publish(self, organism_type, whycon_coords):
          bio_location = Biolocation()
          bio_location.organism_type = organism_type
          bio_location.whycon_x = whycon_coords[0]
          bio_location.whycon_y = whycon_coords[1]
          bio_location.whycon_z = whycon_coords[2]

          self.location_pub.publish(bio_location)

          rospy.loginfo(f"Published {organism_type} at coordinates: {whycon_coords}")

          



        def search_and_identify(self):
          rate = rospy.Rate(10)  
          while not rospy.is_shutdown():
              reached_waypoint = self.navigate_to_waypoint()

              if reached_waypoint:
                  rospy.loginfo("Search grid completed. Heading to research station.")
                  break

              
              rate.sleep()

          self.go_to_research_station()


        def compute_error(self):
            self.alt_error = self.setpoint[2] - self.current_pose[2]
            self.roll_error = self.setpoint[0] - self.current_pose[0]
            self.pitch_error = self.setpoint[1] - self.current_pose[1]

            self.sum_alt_error += self.alt_error
            self.sum_roll_error += self.roll_error
            self.sum_pitch_error += self.pitch_error

        def update_command(self):
            self.cmd.rcThrottle = int(1500 + self.alt_error * self.Kp[2] +
                                      self.sum_alt_error * self.Ki[2] +
                                      (self.alt_error - self.prev_alt_error) * self.Kd[2])
            self.cmd.rcRoll = int(1500 - self.roll_error * self.Kp[0] -
                                  self.sum_roll_error * self.Ki[0] -
                                  (self.roll_error - self.prev_roll_error) * self.Kd[0])
            self.cmd.rcPitch = int(1500 + self.pitch_error * self.Kp[1] +
                                   self.sum_pitch_error * self.Ki[1] +
                                   (self.pitch_error - self.prev_pitch_error) * self.Kd[1])

            self.cmd.rcThrottle = max(1000, min(self.cmd.rcThrottle, 2000))
            self.cmd.rcRoll = max(1000, min(self.cmd.rcRoll, 2000))
            self.cmd.rcPitch = max(1000, min(self.cmd.rcPitch, 2000))

            self.prev_alt_error = self.alt_error
            self.prev_roll_error = self.roll_error
            self.prev_pitch_error = self.pitch_error

        def navigate_to_waypoint(self):
            self.setpoint = self.waypoints[self.current_waypoint_idx]
            self.compute_error()
            self.update_command()
            self.command_pub.publish(self.cmd)

            threshold = 0.2  
            if all(abs(self.current_pose[i] - self.setpoint[i]) < threshold for i in range(3)):
                if self.current_waypoint_idx < len(self.waypoints) - 1:
                    self.current_waypoint_idx += 1



    def go_to_research_station(self):
      research_station_coords = [11, 11, 37]

      if self.current_pose is None:
          rospy.logerr("Current pose not available.")
          return

      while not rospy.is_shutdown():
          dx = research_station_coords[0] - self.current_pose[0]
          dy = research_station_coords[1] - self.current_pose[1]
          dz = research_station_coords[2] - self.current_pose[2]

          
          threshold = 0.5  
          if abs(dx) < threshold and abs(dy) < threshold and abs(dz) < threshold:
              rospy.loginfo("Drone has arrived at the research station.")
              break

          Kp = 0.5
          self.cmd.rcPitch = int(1500 + Kp * dy)
          self.cmd.rcRoll = int(1500 - Kp * dx)
          self.cmd.rcThrottle = int(1500 + Kp * dz)

          self.cmd.rcPitch = max(1000, min(self.cmd.rcPitch, 2000))
          self.cmd.rcRoll = max(1000, min(self.cmd.rcRoll, 2000))
          self.cmd.rcThrottle = max(1000, min(self.cmd.rcThrottle, 2000))

          self.command_pub.publish(self.cmd)
          rospy.sleep(1)

      self.cmd.rcThrottle = 1000  
      self.command_pub.publish(self.cmd)
      rospy.sleep(2)  
      self.disarm()



if __name__ == '__main__':
    try:
        swift_drone = SwiftDrone()
        r = rospy.Rate(30)  

        while not rospy.is_shutdown():
            swift_drone.navigate_to_waypoint()
            r.sleep()
    except rospy.ROSInterruptException:
        pass
