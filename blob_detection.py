import cv2  
  
# np is an alias pointing to numpy library 
import numpy as np 

im = cv2.imread("images/004_new.jpeg", cv2.IMREAD_GRAYSCALE)

params = cv2.SimpleBlobDetector_Params()

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Change thresholds
# params.minThreshold = 10;
# params.maxThreshold = 200;

# Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01


# Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1

# Filter by Area.
# params.filterByArea = True
# params.minArea = 15

# Filter by Color.
# params.filterByColor = True
# params.blobColor = 255



detector = cv2.SimpleBlobDetector_create(params)
 
# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)