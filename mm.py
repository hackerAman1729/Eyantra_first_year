#!/usr/bin/env python3

# Imports
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

# Class Definition
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

    def generate_search_grid(self):
      grid = []
      # Define the boundary coordinates
      x_min, x_max = -8, 8
      y_min, y_max = -8, 8
      # Define the step size for each grid point, this will depend on the drone's camera field of view and altitude
      step_size = 1.0

      # Generate grid points
      y = y_min
      while y <= y_max:
          x = x_min if (y / step_size) % 2 == 0 else x_max
          while 0 <= x <= x_max if (y / step_size) % 2 == 0 else x_max >= x >= x_min:
              # Append the current grid point
              grid.append((x, y, 20))  # 20 is an arbitrary altitude for the drone
              # Move to the next point in the row
              x += step_size if (y / step_size) % 2 == 0 else -step_size
          # Move to the next row
          y += step_size

      return grid


    def init_command(self):
      self.cmd = swift_msgs()
      # Set all RC commands to neutral midpoint values
      self.cmd.rcRoll = 1500
      self.cmd.rcPitch = 1500
      self.cmd.rcYaw = 1500
      self.cmd.rcThrottle = 1500
      # Aux channels can be set to midpoint values or used for specific functions
      # depending on the drone configuration
      self.cmd.rcAUX1 = 1500
      self.cmd.rcAUX2 = 1500
      self.cmd.rcAUX3 = 1500
      self.cmd.rcAUX4 = 1500
    


    def arm(self):
      # Set the throttle to zero and AUX4 (which might be used for arming the drone) to high
      self.cmd.rcThrottle = 1000
      self.cmd.rcAUX4 = 2000  # Assuming 2000 is the value to arm the drone; this could vary
      self.command_pub.publish(self.cmd)
      rospy.sleep(1)  # Give the drone time to arm

      # Set back to neutral after arming (except throttle which remains at minimum)
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
      # Set the AUX4 to a low value to disarm the drone; exact value may vary based on the drone's configuration
      self.cmd.rcAUX4 = 1000
      # It's a good practice to also set throttle to minimum when disarming for safety
      self.cmd.rcThrottle = 1000
      self.command_pub.publish(self.cmd)
      rospy.sleep(1)  # Give the drone time to disarm

      # Reset all other controls to neutral
      self.cmd.rcRoll = 1500
      self.cmd.rcPitch = 1500
      self.cmd.rcYaw = 1500
      self.cmd.rcAUX1 = 1500
      self.cmd.rcAUX2 = 1500
      self.cmd.rcAUX3 = 1500
      # Keep AUX4 at disarm value
      self.command_pub.publish(self.cmd)
      rospy.sleep(1)
    


    def pose_callback(self, msg):
    # Assuming the message contains the pose of the drone as the first element in the poses array
     if msg.poses:
        pose = msg.poses[0]
        self.current_pose = (pose.position.x, pose.position.y, pose.position.z)


    def image_callback(self, data):
      try:
        cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        rospy.logerr(e)
        return

      # Image processing to detect LEDs
      gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
      blurred = cv2.GaussianBlur(gray, (11, 11), 0)
      thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
      thresh = cv2.erode(thresh, None, iterations=2)
      thresh = cv2.dilate(thresh, None, iterations=4)

      # Label distinct areas in the image
      labels = measure.label(thresh, background=0)
      mask = np.zeros(thresh.shape, dtype="uint8")

      # Find and process each area
      for label in np.unique(labels):
        # Ignore the background
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # If the area is large enough, consider it an LED (organism)
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)

      # Find the contours in the mask
      cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)

      # Proceed if at least one contour was found
      if cnts:
        # Find the largest contour which will be the organism
        c = max(cnts, key=cv2.contourArea)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)

        # If the organism is significant enough
        if radius > 10:
            # Draw the contour and centroid on the image
            cv2.drawContours(cv_image, [c], -1, (0, 255, 0), 2)
            cv2.circle(cv_image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)

            # Align the centroid of the LED to the center of the camera frame
            # Calculate the offsets from the center of the image
            center_offset_x = cv_image.shape[1]//2 - cX
            center_offset_y = cv_image.shape[0]//2 - cY

            # Depending on your setup, you might actuate the drone to align the LED
            # to the center or use these offsets to calculate the precise location
            # relative to the drone's current pose

            # Example (pseudo-code):
            # self.align_drone(center_offset_x, center_offset_y)

            # For now, just log the offsets
            rospy.loginfo(f"Offset from center: x={center_offset_x}, y={center_offset_y}")
    


    def navigate_to_waypoint(self):
      if self.current_pose is None or self.waypoint_index >= len(self.search_grid):
        # We either don't have a current position or we've finished the search grid
        return False

      # Get the current waypoint based on the drone's position in the search grid
      waypoint = self.search_grid[self.waypoint_index]
      # Calculate the difference between the current pose and the waypoint
      dx = waypoint[0] - self.current_pose[0]
      dy = waypoint[1] - self.current_pose[1]
      dz = waypoint[2] - self.current_pose[2]

      # Determine if we are close enough to the waypoint to consider it 'reached'
      threshold = 0.5  # Threshold distance to waypoint to consider it 'reached'
      if abs(dx) < threshold and abs(dy) < threshold and abs(dz) < threshold:
        # Move to the next waypoint
        self.waypoint_index += 1
        if self.waypoint_index >= len(self.search_grid):
            return True  # All waypoints have been reached
        else:
            return False  # There are more waypoints to navigate to

      # Create a proportional control to head towards the waypoint
      # These gains will need to be tuned to your specific drone dynamics
      Kp = 0.5
      self.cmd.rcPitch = int(1500 + Kp * dy)
      self.cmd.rcRoll = int(1500 - Kp * dx)
      self.cmd.rcThrottle = int(1500 + Kp * dz)

      # Ensure the commands are within the valid range for the drone
      self.cmd.rcPitch = max(1000, min(self.cmd.rcPitch, 2000))
      self.cmd.rcRoll = max(1000, min(self.cmd.rcRoll, 2000))
      self.cmd.rcThrottle = max(1000, min(self.cmd.rcThrottle, 2000))

      # Send the command
      self.command_pub.publish(self.cmd)
      return False
    


    def align_and_publish(self, organism_type, whycon_coords):
      # Create a Biolocation message
      bio_location = Biolocation()
      bio_location.organism_type = organism_type
      bio_location.whycon_x = whycon_coords[0]
      bio_location.whycon_y = whycon_coords[1]
      bio_location.whycon_z = whycon_coords[2]

      # Publish the organism's details
      self.location_pub.publish(bio_location)

      # Log the event
      rospy.loginfo(f"Published {organism_type} at coordinates: {whycon_coords}")

      # Assuming the drone has a mechanism to align itself with the organism based on whycon coordinates,
      # we would call that mechanism here. If that's not the case, you'd have to control the drone to
      # the location of the organism.

      # Since this is simulation-based and we're not provided with the actual alignment mechanism, 
      # we'll consider the drone to be instantly aligned for the purpose of this exercise.
    


    def search_and_identify(self):
      # Run the main loop for the drone operation
      rate = rospy.Rate(10)  # 10 Hz control loop
      while not rospy.is_shutdown():
          # Navigate to the current waypoint
          reached_waypoint = self.navigate_to_waypoint()

          # If all waypoints have been reached, break from the loop
          if reached_waypoint:
              rospy.loginfo("Search grid completed. Heading to research station.")
              break

          # The image_callback function will handle organism detection and alignment
          # The pose_callback function will update the current drone position

          # Sleep to maintain the loop rate
          rate.sleep()

      # Once the search grid is completed, navigate to the research station
      self.go_to_research_station()
    


def go_to_research_station(self):
  # Define the research station coordinates
  research_station_coords = [11, 11, 37]

  # Check if we have the current pose
  if self.current_pose is None:
      rospy.logerr("Current pose not available.")
      return

  # Fly towards the research station
  while not rospy.is_shutdown():
      # Calculate the differences to the research station
      dx = research_station_coords[0] - self.current_pose[0]
      dy = research_station_coords[1] - self.current_pose[1]
      dz = research_station_coords[2] - self.current_pose[2]

      # Check if the drone is close enough to the research station to be considered arrived
      threshold = 0.5  # Threshold distance to consider the drone has arrived
      if abs(dx) < threshold and abs(dy) < threshold and abs(dz) < threshold:
          rospy.loginfo("Drone has arrived at the research station.")
          break

      # Use proportional control to fly towards the research station
      Kp = 0.5
      self.cmd.rcPitch = int(1500 + Kp * dy)
      self.cmd.rcRoll = int(1500 - Kp * dx)
      self.cmd.rcThrottle = int(1500 + Kp * dz)

      # Ensure the commands are within the drone's valid range
      self.cmd.rcPitch = max(1000, min(self.cmd.rcPitch, 2000))
      self.cmd.rcRoll = max(1000, min(self.cmd.rcRoll, 2000))
      self.cmd.rcThrottle = max(1000, min(self.cmd.rcThrottle, 2000))

      # Publish the command
      self.command_pub.publish(self.cmd)
      rospy.sleep(1)

  # Land the drone
  self.cmd.rcThrottle = 1000  # Assuming 1000 is the minimum throttle value for landing
  self.command_pub.publish(self.cmd)
  rospy.sleep(2)  # Wait for the drone to land

  # Disarm the drone after landing
  self.disarm()


# Main function
if __name__ == '__main__':
    try:
        swift_drone = SwiftDrone()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
