# task 1b

from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2

# Image loading using imread
image = cv2.imread('led.jpeg', 1)

# converting the image to greyscale and blurring it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

#Contour sorting
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

num_leds = 0
centroid_list = []
area_list = []



# Contour looping
for (i, c) in enumerate(cnts):
    area = cv2.contourArea(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (int(cX), int(cY)), 3, (0, 0, 255), -1)
    centroid_list.append((cX, cY))
    area_list.append(area)
    num_leds += 1

cv2.imwrite("led_detection_results.png", image)

with open("led_detection_results.txt", "w") as file:
    file.write(f"No. of LEDs detected: {num_leds}\n")
    for i, (centroid, area) in enumerate(zip(centroid_list, area_list)):
        file.write(f"Centroid #{i + 1}: {centroid}\nArea #{i + 1}: {area}\n")