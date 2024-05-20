import cv2
import numpy as np
import math

image = cv2.imread('/home/theta/rins/src/dis_tutorial3/maps/map.pgm')
original_image = image.copy()

# convert the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# threshold the image
_, image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)

dst = cv2.Canny(image, 50, 200, None, 3)

# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv2.HoughLines(dst, 1, np.pi / 180, 25, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1, pt2, (0,0,0), 3, cv2.LINE_AA)

# erode
kernel = np.ones((10,10), np.uint8)
image = cv2.erode(image, kernel, iterations=1)

cv2.imshow("Image", image)
cv2.waitKey(0)

# find connected components and their centroids
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

# print the centroids
for i in range(1, num_labels):
    print(f"Centroid {i}: {centroids[i]}")

# display the centroids
for i in range(1, num_labels):
    cv2.circle(original_image, (int(centroids[i][0]), int(centroids[i][1])), 5, (0, 255, 0), -1)

cv2.imshow("Centroids", original_image)
cv2.waitKey(0)